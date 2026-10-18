"""Associate completed videos with their exact masters for automatic resume."""

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time

import torch

import folder_paths
from comfy.cli_args import args
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import Types, io, ui
from comfy_extras.nodes_video import SaveVideo

from .latent import _frame_at_latent as latents_to_frames

AUTO = "auto (latest completed for this project)"


_SLUG_BAD = re.compile(r'[\\/:*?"<>|%\x00-\x1f]+')


def output_slug(name, limit=96):
    """Folder/file-safe form of a project or variant name: path separators, `..`, format
    placeholders (`%`) and control characters are dropped; spaces are kept like the master names."""
    text = _SLUG_BAD.sub("_", str(name or ""))
    text = re.sub(r"\s+", " ", text)
    while ".." in text:
        text = text.replace("..", "_")
    return text.strip(" ._")[:limit].rstrip(" ._")


def run_filename_prefix(filename_prefix, project, variant=""):
    """`final_vids/video` + project P + variant V -> `final_vids/P/video_V`.

    The widget's folder and basename are kept; the project becomes its own folder and the variant
    a basename suffix, so every project/variant counts from its own 00001 and two pods rendering
    different projects can never be handed the same file name.
    """
    folder, base = os.path.split(str(filename_prefix or "video").strip().strip("/"))
    base = base or "video"
    parts = [folder] if folder else []
    project_slug = output_slug(project)
    if project_slug:
        parts.append(project_slug)
    variant_slug = output_slug(variant)
    parts.append(f"{base}_{variant_slug}" if variant_slug else base)
    return "/".join(parts)


def reserve_output_file(full_output_folder, filename, counter, extension, attempts=100000):
    """Create `<filename>_<counter>_.<extension>` exclusively and return (file, counter).

    O_CREAT|O_EXCL is atomic on the shared filesystem: whichever process reserves a number second
    gets FileExistsError and moves on to the next one. The empty placeholder is what the other
    pod's listing sees until the encoder truncates it with real data, so the same number is never
    chosen twice even when both pods list the folder in the same second.
    """
    os.makedirs(full_output_folder, exist_ok=True)
    for _ in range(attempts):
        file = f"{filename}_{counter:05}_.{extension}"
        path = os.path.join(full_output_folder, file)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            counter += 1
            continue
        os.close(fd)
        return file, counter
    raise RuntimeError(f"No free output name for {filename!r} in {full_output_folder}")


def _discard_reserved(path):
    """A failed or interrupted encode must not leave a partial file under a valid name."""
    try:
        os.unlink(path)
    except OSError:
        pass


def video_fingerprint(path):
    stat = os.stat(path)
    return {"video_size": int(stat.st_size), "video_mtime_ns": int(stat.st_mtime_ns)}


def _stamp(ns):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ns / 1e9))


def video_changed(record):
    """Why the record's video can no longer be trusted, or None when it is the recorded render.

    Records carry the finished file's size and mtime (since 2026-09-18); an older record without
    them is checked by frame count (one ffprobe). A file that was rewritten under the same name by
    another run fails either check and is refused instead of feeding foreign frames to the prefix.
    The mtime is compared at whole seconds (the shared filesystem stores seconds); a size-identical
    file with a different mtime -- a restored copy -- is accepted when its frame count matches.
    """
    video = record["video"]
    try:
        stat = os.stat(video)
    except OSError:
        return "the video file is missing"
    if "video_size" in record and "video_mtime_ns" in record:
        if stat.st_size != int(record["video_size"]):
            return (f"the video was rewritten after this run finished (recorded {record['video_size']} bytes "
                    f"at {_stamp(record['video_mtime_ns'])}, on disk {stat.st_size} bytes at {_stamp(stat.st_mtime_ns)})")
        if stat.st_mtime_ns // 10**9 == int(record["video_mtime_ns"]) // 10**9:
            return None
    frames = probe_frame_count(video)
    if frames != int(record["frames"]):
        return (f"the video holds {'an unreadable number of' if frames is None else frames} frames "
                f"while this run saved {record['frames']} (rewritten by another run?)")
    return None


def _index_directory():
    return Path(folder_paths.get_output_directory()) / "latents" / "h3_resume_runs"


# Small JSON metadata only. Every lookup rechecks record identity and selected files.
_RECORD_CACHE = {}


def _run_records():
    directory = _index_directory()
    records, present = [], set()
    if directory.exists():
        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.name.endswith('.json') or not entry.is_file():
                    continue
                path = entry.path
                present.add(path)
                stat = entry.stat()
                signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                cached = _RECORD_CACHE.get(path)
                if cached is None or cached[0] != signature:
                    record = json.loads(Path(path).read_text())
                    _RECORD_CACHE[path] = (signature, record)
                else:
                    record = cached[1]
                records.append(record)
    for path in _RECORD_CACHE.keys() - present:
        del _RECORD_CACHE[path]
    return sorted(records, key=lambda r: r['completed_at_ns'], reverse=True)


def completed_runs():
    return [r.copy() for r in _run_records()
            if Path(r['master']).is_file() and Path(r['video']).is_file()]


def record_run(project, variant, master_path, video_path, frames, fps, width, height,
               completed_at_ns=None):
    master = Path(master_path).resolve()
    video = Path(video_path).resolve()
    if not master.is_file() or not video.is_file():
        raise FileNotFoundError("Both the master and video must finish saving before this run can be resumed.")
    # Read metadata/shapes without loading the tensor storage into RAM.
    saved = torch.load(master, map_location="cpu", weights_only=True, mmap=True)
    if int(saved.get("start_chunk", 0)) != 0:
        raise ValueError("Automatic prefix resume requires a full master starting at chunk 0.")
    master_frames = latents_to_frames(saved["video"].shape[2])
    if master_frames != int(frames):
        raise ValueError(f"Master/video length mismatch: {master_frames} versus {frames} frames.")
    record = {
        "project": str(project), "variant": str(variant or ""),
        "master": str(master), "video": str(video),
        "frames": int(frames), "fps": float(fps),
        "width": int(width), "height": int(height),
        "chunk_frames": int(saved.get("chunk_frames", 0)),
        "overlap_frames": int(saved.get("overlap_frames", 0)),
        "completed_at_ns": int(completed_at_ns if completed_at_ns is not None else time.time_ns()),
        **video_fingerprint(video),
        "pod": socket.gethostname(),
    }
    directory = _index_directory()
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(master).encode()).hexdigest()
    destination = directory / f"{key}.json"
    temporary = directory / f"{key}.json.tmp"
    temporary.write_text(json.dumps(record, indent=2))
    os.replace(temporary, destination)
    return record


def resolve_run(project, variant, master):
    selected = None if master == AUTO else str(Path(master).resolve())
    for record in _run_records():
        if record['project'] != project or record['variant'] != (variant or ''):
            continue
        if selected is not None and record['master'] != selected:
            continue
        if not (Path(record['master']).is_file() and Path(record['video']).is_file()):
            continue
        problem = video_changed(record)
        if problem:
            # Never fall through to an older run silently and never load foreign frames: the
            # user decides. The master latent is untouched by a video overwrite.
            raise ValueError(
                f"The {'selected' if selected else 'latest completed'} run for this project/variant cannot be "
                f"resumed: {problem}.\n  master: {record['master']}\n  video:  {record['video']}\n"
                "The master latent is intact: decode it and save the result through HGSaveResumeVideo "
                "to rebuild the pair, or choose an older master in MMH3 Resume Run.")
        return record.copy()
    if selected is None:
        raise ValueError("No completed run exists for this project and prompt variant. Set RESUME C to 0 for the first run.")
    raise ValueError("The selected master has no completed video pair for this project and prompt variant. Choose auto or a matching saved run.")






def probe_frame_count(path):
    """Video-stream frame count via ffprobe; None when the file is unreadable or has no index yet."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = probe.stdout.strip().splitlines()
    value = lines[0].strip() if lines else ""
    return int(value) if value.isdigit() and int(value) > 0 else None






class HGLoadResumeRun(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HGLoadResumeRun", display_name="MMH3 Resume Run",
            category="Hybrid Guides",
            description="Load the latest completed master and its matching video for this project, or select an older master.",
            inputs=[
                io.String.Input("project", force_input=True),
                io.Int.Input("resume_from", default=0, min=0, force_input=True),
                io.Combo.Input("master", options=[AUTO] + [r["master"] for r in completed_runs()], default=AUTO,
                               tooltip="Auto selects the latest completed run for the wired project/variant. Refresh ComfyUI to list newly saved masters."),
                io.String.Input("variant", default="", optional=True, force_input=True),
            ],
            outputs=[io.Latent.Output("latent"), io.Int.Output("latent_t"),
                     io.Int.Output("frames"), io.Int.Output("start_chunk"),
                     io.String.Output("master_path"), io.String.Output("video_path"),
                     io.String.Output("report")],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, project, resume_from, master=AUTO, variant=""):
        if int(resume_from) == 0:
            return io.NodeOutput(None, 0, 0, 0, "", "", "Fresh run; no saved master or video loaded.")
        record = resolve_run(project, variant, master)
        saved = torch.load(record["master"], map_location="cpu", weights_only=True)
        video, audio = saved["video"].float(), saved["audio"].float()
        meta = {k: int(saved[k]) for k in ("start_chunk", "chunk_frames", "overlap_frames") if k in saved}
        latent = {"samples": NestedTensor([video, audio]), "h3_meta": meta}
        for key in ("h3_warmup",):
            if key in saved:
                latent[key] = saved[key]
        prefix_video = record["video"]
        report = (f"Resume {project}" + (f" / {variant}" if variant else "") +
                  f": {record['frames']} saved frames; accepting {int(resume_from)} chunks.\n"
                  f"Master: {record['master']}\nVideo: {prefix_video}")
        return io.NodeOutput(latent, int(video.shape[2]), record["frames"],
                             int(saved.get("start_chunk", 0)), record["master"], prefix_video, report)


class HGSaveResumeVideo(SaveVideo):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "HGSaveResumeVideo"
        schema.display_name = "MMH3 Save Video and Resume Pair"
        schema.description = ("Save the video under <prefix folder>/<project>/<prefix name>[_<variant>]_NNNNN_ with the "
                              "name reserved exclusively before encoding (two pods sharing the output folder can never "
                              "overwrite each other), then remember its exact master and file fingerprint for automatic resume.")
        schema.category = "Hybrid Guides"
        schema.inputs.extend([
            io.String.Input("master_path", force_input=True),
            io.String.Input("project", force_input=True),
            io.String.Input("variant", default="", optional=True, force_input=True),
        ])
        schema.outputs.extend([io.String.Output("master_path"), io.String.Output("video_path"), io.String.Output("report")])
        return schema

    @classmethod
    def execute(cls, video, filename_prefix, format, master_path, project, codec=None, variant=""):
        # Core SaveVideo.execute, with two changes: the prefix is namespaced per project/variant
        # and the file name is reserved exclusively BEFORE encoding (see the module header).
        if isinstance(format, dict):
            format_name = format["format"]
            codec = format.get("codec") or codec
        else:
            format_name = format
        if codec is None:
            codec = {"codec": "auto"}
        codec_name = codec["codec"]
        if format_name == "auto":
            format_name = "webm" if codec_name == "av1" else "mp4"
        encoding = codec.get("encoding") or {}
        width, height = video.get_dimensions()
        prefix = run_filename_prefix(filename_prefix, project, variant)
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory(), width, height)
        extension = Types.VideoContainer.get_extension(format_name)
        file, counter = reserve_output_file(full_output_folder, filename, counter, extension)
        path = os.path.join(full_output_folder, file)
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            hidden = cls.hidden
            if hidden is not None and hidden.extra_pnginfo is not None:
                metadata.update(hidden.extra_pnginfo)
            if hidden is not None and hidden.prompt is not None:
                metadata["prompt"] = hidden.prompt
            if metadata:
                saved_metadata = metadata
        try:
            video.save_to(path, format=Types.VideoContainer(format_name), codec=Types.VideoCodec(codec_name),
                          metadata=saved_metadata, crf=encoding.get("crf"))
        except BaseException:
            _discard_reserved(path)
            raise
        record = record_run(project, variant, master_path, path,
                            video.get_frame_count(), video.get_frame_rate(), width, height)
        report = (f"Saved {record['frames']} frames as {os.path.join(subfolder, file)} "
                  f"(name reserved exclusively; each project/variant counts in its own folder). "
                  f"This master/video pair is available for auto resume.")
        return io.NodeOutput(video, record["master"], record["video"], report,
                             ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))


class HGResumePlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HGResumePlan", display_name="H3 Resume Plan",
            category="Hybrid Guides",
            description="Plan a fresh run or resume from accepted chunks, including the pinned window and source offsets.",
            inputs=[
                io.Int.Input("total_frames", force_input=True, min=5),
                io.Int.Input("chunk_frames", force_input=True, min=5),
                io.Int.Input("overlap_frames", force_input=True, min=0),
                io.Int.Input("accepted_chunks", default=0, min=0,
                             tooltip="Completed chunks to keep. 0 starts fresh; a smaller count redoes an earlier suffix."),
                io.Int.Input("new_chunks", default=1, min=0,
                             tooltip="New chunks to generate. 0 renders the remaining project."),
                io.Float.Input("fps", default=24., min=.001),
            ],
            outputs=[
                io.Int.Output("chunk_count"), io.Int.Output("run_start"),
                io.Float.Output("run_start_seconds"), io.Int.Output("run_length"),
                io.Int.Output("windows_this_run"), io.Int.Output("start_window"),
                io.Int.Output("resuming"), io.Int.Output("accepted_prefix_frames"),
                io.Int.Output("pinned_frames"), io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, total_frames, chunk_frames, overlap_frames, accepted_chunks=0,
                new_chunks=1, fps=24.):
        stride = chunk_frames - overlap_frames
        if stride <= 0:
            raise ValueError("Overlap must be smaller than chunk_frames.")
        count = 1 + math.ceil(max(total_frames - chunk_frames, 0) / stride)
        if accepted_chunks < 0 or accepted_chunks >= count:
            raise ValueError(f"Accept fewer than {count} chunks so there is a continuation to render.")
        resuming = int(accepted_chunks > 0)
        start_window = max(accepted_chunks - 1, 0)
        fresh = count - accepted_chunks
        if new_chunks > 0:
            fresh = min(new_chunks, fresh)
        windows = fresh + resuming
        start = start_window * stride
        length = min(chunk_frames + (windows - 1) * stride, total_frames - start)
        accepted_frames = accepted_chunks * stride + overlap_frames if resuming else 0
        pinned = chunk_frames if resuming else 0
        report = (f"Keep {accepted_chunks} chunks / {accepted_frames} frames; generate {fresh} new chunks. "
                  f"Source frames {start}–{start + length - 1}; {windows} windows including {resuming} pinned. "
                  f"Next accepted chunk count: {accepted_chunks + fresh}.")
        return io.NodeOutput(count, start, start / fps, length, windows, start_window,
                             resuming, accepted_frames, pinned, report)
