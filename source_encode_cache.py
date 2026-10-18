"""HGCachedSourceEncode -- the v2v SOURCE encode with a per-clip disk cache and a
LAZY `images` input.

Why: in the v4 inpaint flows the source video is VAE-encoded on EVERY part of a
project, and a resume part used to load + encode everything from frame 0 through
its own end (2026-09-05 measurement: ~2.3 s per 17-frame clip at 1344x768 on the
H200; a C=11 part encoded 242 clips to use 122). The H3 video VAE encodes each
17-frame clip INDEPENDENTLY (no carried state; see MMH3Tools' MMH3StreamingEncode
docstring), so a clip's latent can be cached on disk and re-assembled exactly.

What it does (drop-in for MMH3StreamingEncode on the source path):
  * same math as MMH3StreamingEncode, verified bit-identical: per pass
    process_input -> vae dtype -> [1,3,T,H,W] -> pixel mean/std -> _adaptive_encode
    per clip; token_drop once at the end; latents mean/std normalisation;
  * every clip's moments-mean is written to
    output/<cache_dir>/<key>/clip_<global clip index>.pt (fp16, ~1 MB at 1344x768),
    key = sha1(abs path, mtime_ns, size, WxH, fps). The global clip index is
    start_frame // 17, so parts that start at different run starts SHARE clips as
    long as start_frame is a multiple of 17 (run starts are: stride = chunk -
    overlap is a multiple of 17 on the 17k+5 grid). A misaligned start falls back
    to a window-private key (still cached, just not shared);
  * `images` is LAZY: check_lazy_status only requests it when a needed clip is
    missing, so a full cache hit never executes the VHS loader at all -- no ffmpeg
    decode, no 50 GB IMAGE batch in RAM;
  * to make that skip possible the SOURCE AUDIO comes from this node too (`audio`
    output), extracted with VHS's own get_audio(path, start_s, frames/fps) -- the
    exact call VHS_LoadVideoFFmpegPath makes for the same window, so
    VAEEncodeAudio sees identical samples.

Source file changes invalidate the executor and disk caches. A padded tail clip
(frame count not a multiple of 17) uses a _padN name for that exact range. Direct
source loading is opt-in, for an unmodified VHS source loader: it reads only the
missing spans and keeps the original run seek anchor for identical frame selection.
"""
from contextlib import closing
import hashlib
import json
import itertools
import logging
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import torch

import folder_paths
import comfy.model_management

_LOG = "[HGCachedSourceEncode]"
INDEX_LIMIT = 2 ** 31          # reflect-pad / CUDA kernels need 32-bit addressing


def _h3_video_vae(vae):
    m = getattr(vae, "first_stage_model", None)
    need = ("clip_length", "token_drop", "_adaptive_encode", "pixel_mean", "pixel_std",
            "latents_mean", "latents_std")
    if m is not None and all(hasattr(m, a) for a in need):
        return m
    return None


def _resolve_out(path):
    path = (path or "").strip() or "latents/source_cache"
    if not os.path.isabs(path):
        path = os.path.join(folder_paths.get_output_directory(), path)
    return path


def _vhs_get_audio():
    """VHS's own get_audio (the function VHS_LoadVideoFFmpegPath uses), if loaded."""
    for name, mod in list(sys.modules.items()):
        if name.endswith("videohelpersuite.utils") and hasattr(mod, "get_audio"):
            return mod.get_audio
    return None


def _ffmpeg_exe():
    p = os.environ.get("VHS_FORCE_FFMPEG_PATH")
    if p:
        return p
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _get_audio_fallback(file, start_time=0, duration=0):
    """Vendored copy of VHS utils.get_audio (same ffmpeg args -> same samples)."""
    args = [_ffmpeg_exe(), "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        res = subprocess.run(args + ["-f", "f32le", "-"], capture_output=True, check=True)
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        match = re.search(", (\\d+) Hz, (\\w+), ", res.stderr.decode("utf-8", "ignore"))
    except subprocess.CalledProcessError as e:
        raise Exception("failed to extract audio from %s:\n%s"
                        % (file, e.stderr.decode("utf-8", "ignore")))
    if match:
        ar = int(match.group(1))
        ac = {"mono": 1, "stereo": 2}[match.group(2)]
    else:
        ar, ac = 44100, 2
    audio = audio.reshape((-1, ac)).transpose(0, 1).unsqueeze(0)
    return {"waveform": audio, "sample_rate": ar}


def cache_key(path, width, height, fps):
    st = os.stat(path)
    raw = "%s|%d|%d|%dx%d|%g" % (os.path.abspath(path), st.st_mtime_ns, st.st_size,
                                  int(width), int(height), float(fps))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def clip_plan(start_frame, n_frames, clip):
    """-> (aligned, pad, [(local_index, filename)]) for the window's clips."""
    start_frame = int(start_frame)
    pad = (-n_frames) % clip
    n_clips = (n_frames + pad) // clip
    aligned = start_frame % clip == 0
    base = start_frame // clip
    names = []
    for i in range(n_clips):
        tag = "%06d" % (base + i) if aligned else "w%d_%06d" % (start_frame, i)
        if pad and i == n_clips - 1:
            tag += "_pad%d" % pad
        names.append((i, "clip_%s.pt" % tag))
    return aligned, pad, names


def _source_passes(path, width, height, fps, start_frame, offset, count, batch_size):
    for name, module in tuple(sys.modules.items()):
        if name.endswith('videohelpersuite.load_video_nodes'):
            generator = module.ffmpeg_frame_generator(
                path, fps, count, start_frame / fps, width, height, downscale_ratio=1, skip_frames=offset)
            info = next(generator)
            channels = 4 if info[-1] else 3
            try:
                for start in range(0, count, batch_size):
                    expected = min(batch_size, count - start)
                    frames = np.fromiter(itertools.islice(generator, expected),
                                         dtype=np.dtype((np.float32, (info[8], info[7], channels))))
                    yield offset + start, torch.from_numpy(frames), expected
                    if len(frames) < expected:
                        return
            finally:
                generator.close()
            return
    raise RuntimeError('Load Video Helper Suite to read missing source-cache ranges.')


def missing_passes(indices, clips_per_pass):
    spans = []
    for index in sorted(indices):
        if spans and index == spans[-1][1] and spans[-1][1] - spans[-1][0] < clips_per_pass:
            spans[-1] = (spans[-1][0], index + 1)
        else:
            spans.append((index, index + 1))
    return spans


class HGCachedSourceEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "The H3 VIDEO vae."}),
                "source_path": ("STRING", {"forceInput": True, "tooltip": "The source video file (Get_path_source). Used for the cache key, audio, and optional direct loading of missing frame ranges."}),
                "width": ("INT", {"forceInput": True, "default": 1344, "min": 16, "max": 16384, "tooltip": "Generation width the loader scales to (Get_width) -- part of the cache key."}),
                "height": ("INT", {"forceInput": True, "default": 768, "min": 16, "max": 16384}),
                "fps": ("FLOAT", {"forceInput": True, "default": 24.0, "min": 1.0, "max": 240.0, "tooltip": "Timeline fps (the loader's force_rate)."}),
                "start_frame": ("INT", {"forceInput": True, "default": 0, "min": 0, "max": 10000000, "tooltip": "Timeline frame the loader starts at (Get_run_start; 0 on fresh runs). Multiples of 17 share clips across parts."}),
                "frame_count": ("INT", {"forceInput": True, "default": 124, "min": 1, "max": 10000000, "tooltip": "Frames the loader loads (its frame_load_cap). On a cache hit this is the only length information, so wire the SAME math node as the loader."}),
                "frames_per_chunk": ("INT", {"default": 85, "min": 17, "max": 1700, "step": 17, "tooltip": "Frames per encode pass (multiple of 17); only bounds memory, the result is identical."}),
                "cache_dir": ("STRING", {"default": "latents/source_cache", "tooltip": "Cache folder, relative to the output directory. One subfolder per source file/dims/fps."}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off = plain MMH3StreamingEncode behaviour (always pulls images, never reads or writes the cache)."}),
            },
            "optional": {
                "load_missing_from_source": ("BOOLEAN", {"default": False, "tooltip": "Read only missing frame spans directly from source_path, with VHS fps/crop/scale. Enable only when images is the unmodified VHS source loader."}),
                "images": ("IMAGE", {"lazy": True, "tooltip": "The loader's frames. LAZY: only pulled (and the loader only executed) when a needed clip is not cached."}),
            },
        }

    RETURN_TYPES = ("LATENT", "AUDIO", "STRING")
    RETURN_NAMES = ("latent", "audio", "report")
    FUNCTION = "encode"
    CATEGORY = "Hybrid Guides"
    DESCRIPTION = "Source video VAE encode with a per-clip disk cache; a cache hit never decodes the video (lazy images) and the source audio is extracted here so the loader can be skipped."

    @classmethod
    def IS_CHANGED(cls, source_path, width, height, fps, **kwargs):
        return cache_key(source_path.strip(), width, height, fps)

    # ---- lazy gate ---------------------------------------------------------
    def check_lazy_status(self, vae=None, source_path=None, width=0, height=0, fps=24.0,
                          start_frame=0, frame_count=0, frames_per_chunk=85,
                          cache_dir="latents/source_cache", enabled=True, images=None, load_missing_from_source=False):
        if images is not None:
            return []
        if not enabled:
            return ["images"]
        if load_missing_from_source:
            return []
        try:
            m = _h3_video_vae(vae)
            path = (source_path or "").strip()
            if m is None or not path or not os.path.isfile(path):
                return ["images"]        # let encode() raise a clear error
            cdir = os.path.join(_resolve_out(cache_dir), cache_key(path, width, height, fps))
            _, _, names = clip_plan(int(start_frame), int(frame_count), int(m.clip_length))
            missing = [fn for _, fn in names if not os.path.isfile(os.path.join(cdir, fn))]
            if missing:
                logging.info("%s %d/%d clips not cached -> pulling images", _LOG, len(missing), len(names))
                return ["images"]
            logging.info("%s all %d clips cached -> loader skipped", _LOG, len(names))
            return []
        except Exception as e:      # never wedge the executor on a bookkeeping error
            logging.warning("%s lazy check failed (%s); pulling images", _LOG, e)
            return ["images"]

    # ---- encode ------------------------------------------------------------
    def encode(self, vae, source_path, width, height, fps, start_frame, frame_count,
               frames_per_chunk=85, cache_dir="latents/source_cache", enabled=True,
               images=None, load_missing_from_source=False):
        m = _h3_video_vae(vae)
        if m is None:
            raise ValueError(
                "HGCachedSourceEncode needs the MiniMax H3 VIDEO vae (got %s)."
                % type(getattr(vae, "first_stage_model", vae)).__name__)
        clip = int(m.clip_length)
        path = (source_path or "").strip()
        if not path or not os.path.isfile(path):
            raise ValueError("HGCachedSourceEncode: source_path %r is not a file" % path)
        width, height = int(width), int(height)
        key = cache_key(path, width, height, fps)
        cdir = os.path.join(_resolve_out(cache_dir), key)

        n_req = int(frame_count)
        if images is not None:
            n = int(images.shape[0])
            if int(images.shape[1]) != height or int(images.shape[2]) != width:
                raise ValueError(
                    "HGCachedSourceEncode: images are %dx%d but width/height say %dx%d -- "
                    "wire the same Get_width/Get_height the loader uses (the cache key "
                    "depends on them)." % (int(images.shape[2]), int(images.shape[1]), width, height))
            if n != n_req:
                logging.warning("%s loader delivered %d frames, frame_count says %d -- "
                                "using the loaded count", _LOG, n, n_req)
        else:
            n = n_req
        if n <= 0:
            raise ValueError("HGCachedSourceEncode: no frames to encode")

        aligned, pad, names = clip_plan(int(start_frame), n, clip)
        use_cache = bool(enabled)
        means = [None] * len(names)
        hits = 0
        if use_cache:
            for i, fn in names:
                p = os.path.join(cdir, fn)
                if os.path.isfile(p):
                    try:
                        means[i] = torch.load(p, map_location="cpu", weights_only=True)
                        hits += 1
                    except Exception as e:
                        logging.warning("%s unreadable cache file %s (%s) -- re-encoding", _LOG, p, e)
                        means[i] = None
        todo = {i for i, _ in names if means[i] is None}
        sparse = images is None and use_cache and load_missing_from_source
        if todo and images is None and not sparse:
            raise RuntimeError(
                "HGCachedSourceEncode: %d clip(s) are not cached but the loader's images "
                "were not provided -- the lazy check should have requested them (is "
                "`images` wired?)." % len(todo))

        encoded = 0
        streamed_frames = 0
        if todo:
            fpc = max(clip, (int(frames_per_chunk) // clip) * clip)
            h, w = height, width
            max_fpc = ((INDEX_LIMIT - 1) // max(1, 3 * h * w) // clip) * clip
            if max_fpc < clip:
                raise ValueError("%dx%d is too large to encode even one %d-frame clip"
                                 % (w, h, clip))
            fpc = min(fpc, max_fpc)
            try:
                mem = int(vae.memory_used_encode((1, 3, min(n, fpc), h, w), vae.vae_dtype))
            except Exception:
                mem = 0
            if getattr(vae, "patcher", None) is not None:
                comfy.model_management.load_models_gpu(
                    [vae.patcher], memory_required=mem,
                    force_full_load=getattr(vae, "disable_offload", False))
            if use_cache:
                os.makedirs(cdir, exist_ok=True)
                info = os.path.join(cdir, "info.json")
                if not os.path.isfile(info):
                    st = os.stat(path)
                    with open(info, "w") as metadata:
                        json.dump({"source": os.path.abspath(path), "mtime_ns": st.st_mtime_ns,
                                   "size": st.st_size, "width": width, "height": height,
                                   "fps": float(fps), "clip": clip}, metadata, indent=1)
            if sparse:
                # One seek and a bounded stream even on a completely cold cache.
                # Reading small cached gaps avoids repeatedly decoding from the run start.
                first, last = min(todo) * clip, min((max(todo) + 1) * clip, n)
                passes = _source_passes(path, width, height, fps, start_frame, first, last - first, fpc)
            else:
                passes = ((first * clip, images[first * clip:min(last * clip, n)],
                           min(last * clip, n) - first * clip)
                          for first, last in missing_passes(todo, fpc // clip))
            with closing(passes), torch.inference_mode():
                for pass_start, sub, expected in passes:
                    if sparse:
                        streamed_frames += len(sub)
                    if len(sub) < expected:
                        n = pass_start + len(sub)
                        aligned, pad, names = clip_plan(int(start_frame), n, clip)
                        means = means[:len(names)]
                    if not len(sub):
                        if not n:
                            raise ValueError('The requested source range is past the end of the video.')
                        break
                    pass_end = pass_start + len(sub)
                    if pass_end == n and pad:
                        sub = torch.cat([sub, sub[-1:].repeat(pad, 1, 1, 1)], dim=0)
                        pass_end += pad
                    clip_ids = list(range(pass_start // clip, pass_end // clip))
                    if not any(ci in todo for ci in clip_ids):
                        continue
                    sub = vae.process_input(sub)              # fp32, elementwise: == whole-batch
                    x = sub.to(vae.vae_dtype).to(vae.device)
                    x = x.movedim(-1, 1).movedim(1, 0).unsqueeze(0).contiguous()
                    x = x.add(1.0).mul_(0.5).sub_(m.pixel_mean.to(x)).div_(m.pixel_std.to(x))
                    for j, ci in enumerate(clip_ids):
                        if ci not in todo:
                            continue
                        z = m._adaptive_encode(x[:, :, j * clip:(j + 1) * clip])
                        mean = z[:, :z.shape[1] // 2].cpu()  # == chunk(moments,2)[0] later
                        means[ci] = mean
                        encoded += 1
                        if use_cache:
                            p = os.path.join(cdir, names[ci][1])
                            tmp = p + ".tmp"
                            torch.save(mean, tmp)
                            os.replace(tmp, p)
                    del x, sub
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    logging.info("%s pass %d-%d encoded (%d/%d clips done)", _LOG,
                                 pass_start, pass_end - 1, encoded, len(todo))
                    if pass_end >= n:
                        break

        moments = torch.cat(means, dim=2)
        if m.token_drop > 0:
            moments = moments[:, :, :-int(m.token_drop)]
        mean = moments.float()
        lm = m.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        ls = m.latents_std.view(1, -1, 1, 1, 1).to(mean)
        out = ((mean - lm) / ls).to(vae.output_device).to(vae.vae_output_dtype())

        # source audio for the same window, VHS's own call (identical samples)
        start_s = float(start_frame) / float(fps)
        dur = n * (1.0 / float(fps))
        ga = _vhs_get_audio() or _get_audio_fallback
        audio = ga(path, start_s, dur)
        wf = audio["waveform"]

        report = (
            "source: %s (key %s) @ %dx%d %gfps%s\n"
            "frames %d-%d (%d) -> %d latents, %d clips: %d cached, %d encoded%s\n"
            "cache: %s\n"
            "audio: %.3fs from %.3fs (%d samples @ %d Hz, %s)"
            % (os.path.basename(path), key, width, height, float(fps),
               "" if enabled else " [cache OFF]",
               int(start_frame), int(start_frame) + n - 1, n, int(out.shape[2]), len(names),
               hits, encoded,
               "" if not pad else " (tail clip padded by %d)" % pad,
               cdir if enabled else "-",
               dur, start_s, int(wf.shape[-1]), int(audio["sample_rate"]),
               "VHS get_audio" if ga is not _get_audio_fallback else "ffmpeg fallback"))
        if sparse:
            report += f"\ninput: {streamed_frames} source frames streamed in bounded batches; full image loader skipped"
        if not aligned:
            report += ("\n! start_frame %d is not a multiple of %d: clips cached under a "
                       "window-private key (no sharing across parts)" % (int(start_frame), clip))
        logging.info("%s %s", _LOG, report.splitlines()[1])
        return ({"samples": out}, audio, report)


NODE_CLASS_MAPPINGS = {"HGCachedSourceEncode": HGCachedSourceEncode}
NODE_DISPLAY_NAME_MAPPINGS = {"HGCachedSourceEncode": "H3 Cached Source Encode (per-clip cache, lazy)"}
