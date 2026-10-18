"""Single-prompt conditioning with cached encodes from public MMH3Tools."""
import hashlib
import logging
import sys

import nodes
import node_helpers
from comfy_api.latest import io

from .reference_cache import ReferenceCache, model_key


def upstream():
    cls = nodes.NODE_CLASS_MAPPINGS.get("MMH3ReferenceMultiPrompt")
    if cls is None:
        raise RuntimeError("Install ComfyUI-MMH3Tools and restart ComfyUI.")
    return sys.modules[cls.__module__]


def cached_refs(cache, vae, audio_vae, width, height, frame_count, ref_image_size,
                ref_images, ref_videos, ref_video_audios, ref_audios, ref_window=None):
    base = upstream()
    items, blocks, groups = [], [], []
    fingerprint = hashlib.sha256()
    if ref_images is not None:
        groups.append((ref_images, None, None, None, ref_images, None))
    for name, frames in (ref_videos or {}).items():
        if frames is None:
            continue
        audio_name = 'ref_video_audio_' + name.rsplit('_', 1)[-1]
        audio = (ref_video_audios or {}).get(audio_name)
        effective = (frames, audio) if ref_window is None else base._window_media(frames, audio, ref_window)
        groups.append((None, {name: frames}, {audio_name: audio}, None, effective, frame_count))
    for name, audio in (ref_audios or {}).items():
        if audio is not None:
            groups.append((None, None, None, {name: audio}, audio, None))
    for images, videos, video_audios, audios, media, count in groups:
        identity = (model_key(vae), model_key(audio_vae), width, height,
                    ref_image_size, count, images is not None, bool(videos), media)
        result = cache.get(identity, lambda: base._build_refs(
            vae, audio_vae, width, height, frame_count, ref_image_size,
            images, videos, video_audios, audios, ref_window=ref_window), vae, audio_vae)
        fingerprint.update(cache.last_key.encode())
        items.extend(result[0])
        blocks.extend(result[1])
    cache.fingerprint = fingerprint.hexdigest()
    return items, blocks


class HGReferenceConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HGReferenceConditioning", display_name="HG Cached Reference Conditioning",
            category="Hybrid Guides",
            description="One prompt for every chunk, with reusable reference encodes and advancing reference-video windows.",
            inputs=[
                io.Clip.Input("clip"), io.Vae.Input("vae"), io.Vae.Input("audio_vae"),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", display_name="total_frames", default=192, min=5, max=100000, step=17),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.String.Input("prompts", multiline=True, dynamic_prompts=True,
                                tooltip="One prompt reused for every chunk. The entire text is one prompt."),
                io.Image.Input("ref_images", optional=True),
                io.Autogrow.Input("ref_videos", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("ref_video"), prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("ref_video_audio"), prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3)),
                io.Audio.Input("audio", optional=True),
                io.Boolean.Input("use_input_audio", default=False),
                io.Boolean.Input("unload_text_encoder", default=True),
                io.Boolean.Input("window_ref_video", default=False, optional=True),
                io.Int.Input("chunk_frames", default=0, min=0, max=100000, optional=True),
                io.Int.Input("overlap_frames", default=0, min=0, max=100000, optional=True),
            ],
            outputs=[io.Custom("MMH3_COND_SET").Output(display_name="cond_set"),
                     io.Latent.Output(display_name="latent"), io.Int.Output(display_name="count"),
                     io.Image.Output(display_name="ref_windows")],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, width, height, length, ref_image_size,
                prompts="", ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None, audio=None, use_input_audio=False, unload_text_encoder=True,
                window_ref_video=False, chunk_frames=0, overlap_frames=0):
        base = upstream()
        latent, frame_count = base._empty_av_latent(width, height, length)
        if use_input_audio:
            if audio is None:
                raise ValueError("Connect the input audio or disable use_input_audio.")
            base._use_input_audio(latent, audio_vae, audio)
        windows = None
        if window_ref_video and any(v is not None for v in (ref_videos or {}).values()):
            if chunk_frames <= 0:
                raise ValueError("Connect the sampler's chunk_frames and overlap_frames for windowed references.")
            windows = base._ref_windows(frame_count, int(chunk_frames), int(overlap_frames))
        cache = ReferenceCache()
        built = []
        for span in windows or [None]:
            items, blocks = cached_refs(cache, vae, audio_vae, width, height, frame_count,
                                       ref_image_size, ref_images, ref_videos, ref_video_audios,
                                       ref_audios, ref_window=span)
            built.append((items, blocks, cache.fingerprint))
        cache.finish()

        text = prompts.strip()
        conds = []
        for items, blocks, fingerprint in built:
            key = (text, fingerprint, str(clip.patcher.patches_uuid), model_key(vae), model_key(audio_vae))
            cond = base._CACHE.get(key)
            if cond is None:
                tokens = clip.tokenize(text, minimax_ref_items=items)
                cond = clip.encode_from_tokens_scheduled(tokens)
                if blocks:
                    cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": blocks})
                if len(base._CACHE) >= base._CACHE_MAX:
                    base._CACHE.pop(next(iter(base._CACHE)))
                base._CACHE[key] = cond
            conds.append(cond)
        if unload_text_encoder:
            base.evict_text_encoder(clip, "HGReferenceConditioning")
        logging.info("[Hybrid Guides] One prompt, %d conditioning windows", len(conds))
        return io.NodeOutput({"conds": conds, "prompts": [text], "fingerprint": cache.fingerprint},
                             latent, len(conds), base._window_preview(ref_videos, windows))
