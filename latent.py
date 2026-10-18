import json
import math
import os
import tempfile
import torch
import folder_paths
from comfy.nested_tensor import NestedTensor
_FPT = (1, 4, 4, 4, 4)

def _frame_at_latent(k):
    k = int(k)
    if k <= 0:
        return 0
    full, rem = divmod(k, 5)
    return full * 17 + sum(_FPT[:rem])

def _f2l(frames):
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2

def _audio_at(k):
    return int(round(_frame_at_latent(k) / 24.0 * 40.0))

def _resolve_out(path):
    path = (path or "").strip()
    if path and not os.path.isabs(path):
        path = os.path.join(folder_paths.get_output_directory(), path)
    return path

def _unpack(latent, name):
    s = latent["samples"]
    if isinstance(s, NestedTensor):
        parts = s.unbind()
        return parts[0], parts[-1]
    raise ValueError(
        f"'{name}' is not an H3 AV latent (NestedTensor video+audio pair). "
        "Wire the MMH3 looping sampler's latent output.")

def _san_project(project):
    """Project string -> filesystem-safe filename fragment."""
    import re
    p = re.sub(r"[\\/:]+", "_", (project or "").strip())
    return p

def _master_stem(filename_prefix, project):
    base = _resolve_out((filename_prefix or "").strip() or "latents/h3_v4_master")
    stem = os.path.basename(base)
    proj = _san_project(project)
    if proj:
        stem = f"{stem}_{proj}"
    return os.path.dirname(base), stem

def _latest_master(folder, stem):
    """Newest <stem>_NNNNN.pt in folder, or None."""
    best, best_n = None, -1
    try:
        entries = os.listdir(folder)
    except OSError:
        return None
    for f in entries:
        if f.startswith(stem + "_") and f.endswith(".pt"):
            try:
                n = int(f[len(stem) + 1:-3])
            except ValueError:
                continue
            if n > best_n:
                best, best_n = os.path.join(folder, f), n
    return best

class HGSaveAVLatent:
    """Persist an H3 AV latent to disk (fp16), e.g. the v4 master after sampling.

    This is what makes 'render N chunks now, continue later' and coarse crash
    recovery possible: H3LoadAVLatent + HGContinueMaster resume from this file.

    `start_chunk` records which GLOBAL chunk the master's first window is —
    0 for a fresh run, the pinned chunk's index for a continuation. Without it
    a chain of part-runs (part 3 resuming from part 2's save) would misread
    the prior's timeline. `project` scopes the filename so auto-latest loading
    never picks up another project's master.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {}),
                "filename_prefix": ("STRING", {"default": "latents/h3_v4_master", "tooltip": "Relative to the ComfyUI output directory. '_<project>' (when wired) plus a _%05d counter and .pt are appended."}),
            },
            "optional": {
                "project": ("STRING", {"forceInput": True, "tooltip": "Scopes the filename per project (wire H3ProjectPaths.project) so auto-latest loading cannot cross projects."}),
                "start_chunk": ("INT", {"forceInput": True, "default": 0, "tooltip": "Global chunk index of this master's first window (0 = fresh run; wire the resume shift). Stored in the file for chained resumes."}),
                "chunk_frames": ("INT", {"forceInput": True, "default": 0, "tooltip": "Stamp the chunk grid into the file so a later resume with changed CHUNK SECONDS/OVERLAP errors loudly instead of misaligning."}),
                "overlap_frames": ("INT", {"forceInput": True, "default": 0}),
                "warmup": ("LATENT", {"tooltip": "Sequential sampler denoised_output from this run. Its native low-resolution video is saved inside the finished master for the next continuation."}),
                "warmup_start_chunk": ("INT", {"forceInput": True, "default": 0, "tooltip": "Global first chunk of this run's warmup span; connect the same run start used by HGMergeMasters.current_start."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Hybrid Guides"
    DESCRIPTION = "Save the finished H3 video/audio master with its timeline offset and optional native low-resolution warmup video in one atomic fp16 .pt file."

    def save(self, latent, filename_prefix, project=None, start_chunk=0,
             chunk_frames=0, overlap_frames=0, warmup=None, warmup_start_chunk=0):
        v, a = _unpack(latent, "latent")
        folder, stem = _master_stem(filename_prefix, project)
        os.makedirs(folder or ".", exist_ok=True)
        prev = _latest_master(folder, stem)
        n = 1
        if prev is not None:
            n = int(os.path.basename(prev)[len(stem) + 1:-3]) + 1
        payload = {"video": v.detach().half().cpu(),
                   "audio": a.detach().half().cpu(),
                   "start_chunk": int(start_chunk)}
        if int(chunk_frames) > 0:
            payload["chunk_frames"] = int(chunk_frames)
            payload["overlap_frames"] = int(overlap_frames)
        if warmup is not None:
            wv, _ = _unpack(warmup, "warmup")
            if (chunk_frames < 5 or overlap_frames < 5 or chunk_frames <= overlap_frames
                    or (chunk_frames - 5) % 17 or (overlap_frames - 5) % 17):
                raise ValueError("Saving native warmup requires the matching H3 chunk/overlap grid (5+17k frames).")
            length = _f2l(chunk_frames)
            stride = length - _f2l(overlap_frames)
            offset = (int(warmup_start_chunk) - int(start_chunk)) * stride
            if (wv.ndim != 5 or wv.shape[:2] != v.shape[:2]
                    or wv.shape[2] < 2 or (wv.shape[2] - 2) % 5
                    or offset < 0 or offset + wv.shape[2] > v.shape[2]):
                raise ValueError("The native warmup span must fit inside this finished master's timeline; check warmup_start_chunk and the sampler output.")
            payload["h3_warmup"] = {
                "video": wv.detach().half().cpu().contiguous(),
                "start_chunk": int(warmup_start_chunk),
                "chunk_frames": int(chunk_frames), "overlap_frames": int(overlap_frames),
            }
        # Publish a complete pair atomically. Exclusive linking also prevents two
        # pods sharing this folder from overwriting the same counter.
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".h3-master-", suffix=".tmp", delete=False) as f:
            temporary = f.name
        try:
            torch.save(payload, temporary)
            while True:
                path = os.path.join(folder, f"{stem}_{n:05d}.pt")
                try:
                    os.link(temporary, path)
                    break
                except FileExistsError:
                    n += 1
        finally:
            os.unlink(temporary)
        mb = os.path.getsize(path) / 1e6
        print(f"[HGSaveAVLatent] {path} ({mb:.0f} MB, {int(v.shape[2])} video "
              f"latents, start_chunk {int(start_chunk)})")
        return (path,)

class HGContinueMaster:
    """Build a continuation master for the MMH3 looping sampler.

    Given a prior render's master latent (chunks 0..C-1), emits a master
    covering chunks C-1..N-1 with chunk C-1's window PINNED (written into the
    head + noise-mask black), so chunk C gets its normal masked-overlap carry
    from real content. The pinned window costs nothing: the looping sampler
    (patched 2026-08-24) detects a fully-preserved window and SKIPS its
    sampling — content is carried as-is. (An unpatched sampler re-samples it
    fully masked, ~one chunk's time, same output to codec noise.)

    Wire index_shift into the cond-build loop's index math and remaining_chunks
    into its total: local window j == global chunk (C-1)+j, so prompts and
    control-window starts line up exactly with a fresh full run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prior": ("LATENT", {"tooltip": "The saved master from the earlier run (H3 Load AV Latent)."}),
                "total_frames": ("INT", {"default": 226, "min": 5, "max": 3600, "forceInput": True, "tooltip": "FULL project frames (17j+5) — same value the fresh run uses."}),
                "chunk_frames": ("INT", {"default": 124, "min": 5, "max": 3600, "forceInput": True}),
                "overlap_frames": ("INT", {"default": 39, "min": 0, "max": 3600, "forceInput": True}),
                "completed_chunks": ("INT", {"default": 0, "min": 0, "max": 999, "tooltip": "How many chunks the prior fully rendered (C). 0 = derive from the prior's length. Use a smaller value than derived to REDO from an earlier chunk."}),
                "render_chunks": ("INT", {"default": 0, "min": 0, "max": 9999, "tooltip": "FRESH chunks to render this run (0 or >=remaining = all). The master is sized to the pinned window + this many, so the sampler cannot run past it."}),
            },
            "optional": {
                "prior_start_chunk": ("INT", {"forceInput": True, "default": 0, "tooltip": "Global chunk index of the prior's first window (H3 Load AV Latent's start_chunk output). Makes chained part-runs resolve their timeline correctly."}),
                "source_master": ("LATENT", {"tooltip": "V2V/INPAINT ONLY: the full-project source AV latent (the same MMH3PackAV output a fresh run samples from). The fresh region of the continuation master is filled from its matching span instead of zeros, so denoise_mask 'keep' regions and pinned audio reproduce the SOURCE on resume runs too. Leave unwired for t2v."}),
                "denoise_mask": ("MASK", {"tooltip": "V2V/INPAINT ONLY: the FULL-project inpaint mask (white = regenerate). Emitted re-sliced to this continuation's frame window on the denoise_mask output -- wire that (via the master-select switch) to the sampler, which would otherwise time-squash the full-length mask onto the shorter continuation master. A 1-frame (static) mask passes through unchanged."}),
                "source_start_frame": ("INT", {"forceInput": True, "default": 0, "min": 0, "max": 1000000, "tooltip": "Timeline pixel frame where source_master BEGINS. 0 = the source was loaded from frame 0 (legacy, indexed globally). When the V2V SOURCE loader is windowed to the run span, wire the flow's RUN start (Get_run_start) here so the continuation slices the source relative to that start. Must be a multiple of 17 (a VAE clip boundary) -- run starts are, on the 17k+5 grid."}),
                "mask_start_frame": ("INT", {"forceInput": True, "default": 0, "min": 0, "max": 1000000, "tooltip": "Timeline pixel frame where denoise_mask BEGINS (same idea for a windowed INPAINT MASK loader; 0 = loaded from frame 0). Pixel frames, no clip-boundary requirement."}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING", "MASK")
    RETURN_NAMES = ("latent", "index_shift", "windows_this_run", "report",
                    "denoise_mask")
    FUNCTION = "build"
    CATEGORY = "Hybrid Guides"
    DESCRIPTION = "Continuation master for MMH3 Looping Sampler: pins the last completed chunk's window, renders the rest. Wire index_shift + remaining_chunks into the cond loop."

    def build(self, prior, total_frames, chunk_frames, overlap_frames,
              completed_chunks, render_chunks=0, prior_start_chunk=0,
              source_master=None, denoise_mask=None,
              source_start_frame=0, mask_start_frame=0):
        v, a = _unpack(prior, "prior")
        L = _f2l(int(chunk_frames))
        ov = _f2l(int(overlap_frames)) if int(overlap_frames) >= 5 else 0
        stride = L - ov
        if stride <= 0:
            raise ValueError("HGContinueMaster: overlap must be smaller than chunk_frames")
        # The .pt stamps the grid it was rendered on; every offset below is in
        # chunk units, so a changed CHUNK SECONDS/OVERLAP must refuse loudly
        # (and FIRST — a longer chunk otherwise trips the vaguer too-short
        # error) rather than silently misalign the timeline.
        meta = (prior or {}).get("h3_meta") or {}
        for key, cur in (("chunk_frames", int(chunk_frames)),
                         ("overlap_frames", int(overlap_frames))):
            if meta.get(key) and meta[key] != cur:
                raise ValueError(
                    f"HGContinueMaster: the saved master was rendered with "
                    f"{key}={meta[key]} but this run uses {cur}. Set the widgets "
                    "back, or start fresh — resuming across a grid change would "
                    "misalign every chunk.")

        total_lat = _f2l(int(total_frames))
        n_chunks = 1 + max(0, -(-(total_lat - L) // stride))
        prior_lat = int(v.shape[2])
        if prior_lat < L:
            raise ValueError("HGContinueMaster: the prior holds less than one chunk")

        # The prior's own timeline: it starts at global chunk S (0 for a fresh
        # run; a continuation's save starts at its pinned chunk) and covers
        # W_p windows. Chained part-runs only resolve correctly through S.
        s = max(0, int(prior_start_chunk))
        # ceil: a prior whose LAST window was clamped to the project end still
        # rendered that window; floor would undercount and invite a redo
        w_p = 1 + max(0, -(-(prior_lat - L) // stride))

        c = int(completed_chunks)
        if c <= 0:
            c = s + w_p
        if c <= s:
            raise ValueError(
                f"HGContinueMaster: completed_chunks {c} but the prior only starts "
                f"at chunk {s} -- it cannot supply chunk {c - 1}'s window. Resume "
                "from an earlier master (exact .pt path) to redo that far back.")
        if c > s + w_p:
            raise ValueError(
                f"HGContinueMaster: completed_chunks {c} but the prior covers only "
                f"chunks {s}-{s + w_p - 1}.")
        if c >= n_chunks:
            raise ValueError(
                f"HGContinueMaster: {c} chunks completed of {n_chunks} total -- "
                "nothing left to render.")

        start = (c - 1) * stride              # global, for the new master
        local_start = (c - 1 - s) * stride    # into the prior's tensors
        if prior_lat < local_start + L:
            raise ValueError(
                f"HGContinueMaster: prior has {prior_lat} latents but chunk {c - 1} "
                f"needs through its latent {local_start + L}. Lower completed_chunks.")

        # 1 pinned window + the requested fresh chunks, never past the project end.
        # Without this cap the continuation master covered the WHOLE remaining
        # project regardless of how few chunks were asked for (bit us 2026-08-23:
        # cap=1 + resume C=1 sampled all 29 windows).
        fresh_avail = n_chunks - c
        rc = int(render_chunks)
        fresh = fresh_avail if rc <= 0 else min(rc, fresh_avail)
        windows = 1 + fresh
        cont_lat = min(total_lat - start, (windows - 1) * stride + L)
        cont_f = 17 * ((cont_lat - 2) // 5) + 5
        cont_at = int(round(cont_f / 24.0 * 40.0))

        # The fresh region defaults to ZEROS, which is right for t2v (mask 1 =
        # regenerate everything) and silently wrong for v2v/inpaint: the sampler's
        # denoise_mask "keep" regions reproduce the INPUT latent, so zeros pin
        # grey video and a decoded-zero audio tone (hit 2026-08-24, part 2 of an
        # inpaint run). With source_master wired, the fresh region is the source's
        # own span instead, and keeps reproduce the source on resume runs too.
        src_note = ""
        sv = sa = None
        # Windowed loaders (2026-09-05): the source/mask may begin at a timeline
        # frame > 0 so a resume part loads + encodes only its own span instead of
        # everything from frame 0. The offsets are subtracted from the GLOBAL
        # indices below; 0 keeps the legacy frame-0 indexing bit-for-bit.
        src_off_f = int(source_start_frame or 0)
        if src_off_f < 0 or src_off_f % 17:
            raise ValueError(
                "HGContinueMaster: source_start_frame %d is not a multiple of 17 -- "
                "the source must begin on a VAE clip boundary (run starts are: "
                "stride = chunk - overlap is a multiple of 17 on the 17k+5 grid)."
                % src_off_f)
        src_off = (src_off_f // 17) * 5                     # latents
        if source_master is not None and src_off > start:
            raise ValueError(
                "HGContinueMaster: source_master begins at timeline frame %d "
                "(source_start_frame) but the pinned window starts at frame %d -- "
                "the source loader must start at or before the RUN start."
                % (src_off_f, _frame_at_latent(start)))
        mask_off_f = int(mask_start_frame or 0)
        if mask_off_f < 0:
            raise ValueError("HGContinueMaster: mask_start_frame must be >= 0")
        if source_master is not None:
            sv, sa = _unpack(source_master, "source_master")
            if tuple(sv.shape[3:]) != tuple(v.shape[3:]) or sv.shape[1] != v.shape[1]:
                raise ValueError(
                    "HGContinueMaster: source_master is %s but the prior is %s -- "
                    "channels/spatial dims must match. Wire the SAME packed source "
                    "AV latent the fresh run samples from."
                    % (tuple(sv.shape), tuple(v.shape)))
            if int(sv.shape[2]) + src_off < start + cont_lat:
                raise ValueError(
                    "HGContinueMaster: source_master holds %d latents starting at "
                    "timeline frame %d (latent %d) but this continuation needs "
                    "through global latent %d (frame %d). Load the source through "
                    "the end of this part -- if the loader's frame_load_cap is "
                    "derived from the RENDER CHUNKS cap it must include the resume "
                    "shift: min(total, (shift + cap + pinned - 1) * stride + "
                    "chunk_frames), minus the loader's own start frame when it is "
                    "windowed. (2026-08-24: C=1 cap=3 loaded 651f, needed 855f.)"
                    % (int(sv.shape[2]), src_off_f, src_off, start + cont_lat,
                       17 * ((start + cont_lat - 2) // 5) + 5))

        if sv is not None:
            cv = sv[:, :, start - src_off:start - src_off + cont_lat].float().clone()
        else:
            cv = torch.zeros([v.shape[0], v.shape[1], cont_lat, v.shape[3], v.shape[4]],
                             dtype=torch.float32)
        cv[:, :, :L] = v[:, :, local_start:local_start + L].float()
        ca = torch.zeros([a.shape[0], a.shape[1], a.shape[2], cont_at],
                         dtype=torch.float32)
        if sa is not None:
            sa0 = _audio_at(start - src_off)
            src_at = max(0, min(cont_at, int(sa.shape[-1]) - sa0))
            if src_at > 0:
                ca[:, :, :, :src_at] = sa[:, :, :, sa0:sa0 + src_at].float()
            src_note = ("; fresh region filled from source_master (v2v)"
                        + ("" if src_at >= cont_at else
                           " -- source audio short by %d latents, zero-padded"
                           % (cont_at - src_at)))
        pa0, pa1 = _audio_at(local_start), _audio_at(local_start + L)
        head_at = min(pa1 - pa0, _audio_at(L), int(a.shape[-1]) - pa0, cont_at)
        if head_at > 0:
            ca[:, :, :, :head_at] = a[:, :, :, pa0:pa0 + head_at].float()

        vm = torch.ones([cv.shape[0], 1, cont_lat, cv.shape[3], cv.shape[4]],
                        dtype=torch.float32)
        vm[:, :, :L] = 0.0
        am = torch.ones([ca.shape[0], 1, ca.shape[2], cont_at], dtype=torch.float32)
        am[:, :, :, :max(head_at, 0)] = 0.0

        shift = c - 1
        last = shift + windows - 1
        report = ("continue: %d/%d chunks done (prior covers %d-%d); master = "
                  "chunks %d-%d (%d latents / %d frames), chunk %d PINNED "
                  "(carried; the sampler skips fully-preserved windows), "
                  "chunks %d-%d render fresh%s%s.\n"
                  "index_shift %d, windows %d -- wire both into the cond loop, or "
                  "the prompts/controls land on the wrong chunks. Next run: RESUME "
                  "C = %d."
                  % (c, n_chunks, s, s + w_p - 1, shift, last, cont_lat, cont_f,
                     shift, c, last,
                     "" if fresh == fresh_avail else
                     " (capped; %d more after this run)" % (fresh_avail - fresh),
                     src_note, shift, windows, min(last + 1, n_chunks)))
        if cont_f > 3600:
            report += ("\n! this run's master is %d frames (>3600 / 150s, the pack's "
                       "tested ceiling) -- lower render_chunks to work in parts." % cont_f)

        # The sampler reduces its denoise_mask onto ITS latent's timeline; handed
        # the full-project mask against this SHORTER master it would linspace-squash
        # the whole project's mask into this part (same 2026-08-24 incident). Slice
        # the pixel mask to this continuation's frame window here, where the global
        # offset is known.
        dm_out = None
        if denoise_mask is not None:
            dm = denoise_mask if denoise_mask.ndim == 3 else denoise_mask[None]
            start_f = _frame_at_latent(start)
            if int(dm.shape[0]) <= 1:
                dm_out = dm          # static mask: length-independent as-is
                report += "\ndenoise_mask: static (1 frame), passed through"
            elif mask_off_f > start_f:
                raise ValueError(
                    "HGContinueMaster: denoise_mask begins at timeline frame %d "
                    "(mask_start_frame) but this continuation starts at frame %d -- "
                    "the mask loader must start at or before the RUN start."
                    % (mask_off_f, start_f))
            elif int(dm.shape[0]) < start_f - mask_off_f + cont_f:
                raise ValueError(
                    "HGContinueMaster: denoise_mask has %d frames starting at "
                    "timeline frame %d but this continuation covers global frames "
                    "%d-%d -- load the mask video through the end of this part "
                    "(its frame_load_cap must include the resume shift, like the "
                    "source loader's, minus the loader's own start frame when it "
                    "is windowed)."
                    % (int(dm.shape[0]), mask_off_f, start_f, start_f + cont_f - 1))
            else:
                m0 = start_f - mask_off_f
                dm_out = dm[m0:m0 + cont_f]
                report += ("\ndenoise_mask: sliced to global frames %d-%d%s"
                           % (start_f, start_f + cont_f - 1,
                              "" if not mask_off_f else
                              " (mask loaded from frame %d)" % mask_off_f))

        if sv is not None and src_off_f:
            report += ("\nsource_master: loaded from timeline frame %d (latent %d); "
                       "sliced source-relative latents %d-%d"
                       % (src_off_f, src_off, start - src_off,
                          start - src_off + cont_lat - 1))
        print("[HGContinueMaster] " + report.splitlines()[0])
        return ({"samples": NestedTensor([cv, ca]),
                 "noise_mask": NestedTensor([vm, am]),
                 },
                shift, windows, report, dm_out)

class HGMergeMasters:
    """Splice a resume run's master onto the loaded prior -> full-length latent.

    A continuation master covers only [pinned chunk .. end of this part], so
    decoding it shows just this part. Merging prior[0 : shift*stride] with the
    current master reassembles the whole timeline: chunk boundaries sit on
    stride multiples (grid-safe by construction, unlike joining independent
    clips), and the pinned window in `current` is byte-identical to the same
    span of the prior, so the splice point carries no seam at all.

    Wire: current <- sampler latent, prior <- H3 Load AV Latent, current_start
    <- the resume shift. Fresh runs (shift 0) pass `current` through and the
    prior input is never even pulled (lazy). A LEGACY prior that itself starts
    mid-timeline cannot be spliced to frame 0 -- it passes `current` through
    with a warning instead of failing, so old part-chains keep rendering.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current": ("LATENT", {"tooltip": "This run's sampled master."}),
                "current_start": ("INT", {"forceInput": True, "default": 0, "tooltip": "Global chunk index of current's first window (the resume shift; 0 = fresh run)."}),
                "chunk_frames": ("INT", {"forceInput": True, "default": 124}),
                "overlap_frames": ("INT", {"forceInput": True, "default": 39}),
            },
            "optional": {
                "prior": ("LATENT", {"lazy": True, "tooltip": "The loaded prior master (H3 Load AV Latent). Only pulled on resume runs."}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("latent", "total_latents", "report")
    FUNCTION = "merge"
    CATEGORY = "Hybrid Guides"
    DESCRIPTION = "Reassemble the full-length master from the prior + this resume run, so the output video/save always covers chunk 0 .. end of this part."

    def check_lazy_status(self, current=None, current_start=0, chunk_frames=0,
                          overlap_frames=0, prior=None):
        if int(current_start) > 0 and prior is None:
            return ["prior"]
        return []

    def merge(self, current, current_start, chunk_frames, overlap_frames,
              prior=None):
        shift = int(current_start)
        cv, ca = _unpack(current, "current")
        if shift <= 0:
            return (current, int(cv.shape[2]), "fresh run: passthrough")
        if prior is None:
            raise ValueError(
                "HGMergeMasters: resume run (shift %d) but no prior wired -- "
                "cannot assemble the full timeline." % shift)

        meta = (prior or {}).get("h3_meta") or {}
        if meta.get("start_chunk"):
            pv0, _ = _unpack(prior, "prior")
            msg = ("prior starts at chunk %d (legacy partial save) -- cannot "
                   "splice to frame 0; passing this part through. Final assembly: "
                   "MMH3JoinAV with trim_b_frames = chunk_frames."
                   % meta["start_chunk"])
            print("[HGMergeMasters] ! " + msg)
            return (current, int(cv.shape[2]), "! " + msg)
        for key, cur in (("chunk_frames", int(chunk_frames)),
                         ("overlap_frames", int(overlap_frames))):
            if meta.get(key) and meta[key] != cur:
                raise ValueError(
                    f"HGMergeMasters: prior was rendered with {key}={meta[key]} "
                    f"but this run uses {cur}.")

        L = _f2l(int(chunk_frames))
        ov = _f2l(int(overlap_frames)) if int(overlap_frames) >= 5 else 0
        stride = L - ov
        cut = shift * stride
        pv, pa = _unpack(prior, "prior")
        if int(pv.shape[2]) < cut:
            raise ValueError(
                f"HGMergeMasters: prior has {int(pv.shape[2])} latents but the "
                f"splice needs {cut}. Wrong prior for this resume?")
        if pv.shape[3:] != cv.shape[3:]:
            raise ValueError("HGMergeMasters: prior/current spatial dims differ.")

        a_cut = _audio_at(cut)
        mv = torch.cat([pv[:, :, :cut].float(), cv.float()], dim=2)
        ma = torch.cat([pa[:, :, :, :a_cut].float(), ca.float()], dim=3)
        t = int(mv.shape[2])
        frames = 17 * ((t - 2) // 5) + 5
        report = ("merged: prior chunks 0-%d (%d latents) + this part from chunk "
                  "%d -> full master %d latents / %d frames (%.2fs)"
                  % (shift - 1, cut, shift, t, frames, frames / 24.0))
        print("[HGMergeMasters] " + report)
        return ({"samples": NestedTensor([mv, ma]),
                 "h3_meta": {"start_chunk": 0, "chunk_frames": int(chunk_frames),
                             "overlap_frames": int(overlap_frames)}},
                t, report)
