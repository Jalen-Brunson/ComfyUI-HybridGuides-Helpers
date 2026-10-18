"""Time-gate a mask video: ignore the mask until a timeline frame, then kick in.

Use case: prompt free motion for the first N seconds of an inpaint run, then
start adhering to the matte — frames before the gate become fully-regenerate
(white) so the model is unconstrained there, frames after carry the real mask.
Pure content-space transform on the pixel mask sequence, so everything
downstream (the looping sampler's own mask reduction, HGContinueMaster's
window slicing on RESUME) works unchanged. The mask loader loads from frame 0,
so mask index == timeline frame in the v4 flows, fresh AND resume.
"""

import torch


class HGMaskTimeGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Pixel mask sequence [T,H,W] (frame 0 = timeline frame 0)."}),
                "start_frames": ("INT", {"default": 0, "min": 0, "max": 100000,
                    "tooltip": "Timeline pixel frame where the mask kicks in. 0 = use start_seconds; both 0 = pass-through (node inert)."}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                    "tooltip": "Gate start in seconds (x fps). Used only when start_frames is 0."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0,
                    "tooltip": "Timeline fps for start_seconds."}),
                "before": (["regenerate_all", "keep_source"],
                    {"default": "regenerate_all",
                     "tooltip": "What the mask means BEFORE the gate: regenerate_all = full white "
                                "(model generates freely, the prompted motion); keep_source = full "
                                "black (source pinned everywhere until the gate)."}),
                "fade_frames": ("INT", {"default": 0, "min": 0, "max": 100000,
                    "tooltip": "Cross-fade from the before-state into the real mask over this many frames after the gate start. 0 = hard cut."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "gate"
    CATEGORY = "Hybrid Guides"
    DESCRIPTION = ("Delay a mask video along the timeline: before the gate frame the mask is forced "
                   "to full white (regenerate freely) or full black (pin source); after it the real "
                   "mask applies, optionally cross-faded. Inert at start 0.")

    def gate(self, mask, start_frames, start_seconds, fps, before, fade_frames):
        start = int(start_frames) if start_frames > 0 else int(round(float(start_seconds) * float(fps)))
        if start <= 0:
            return (mask,)
        m = mask
        if m.ndim == 2:
            m = m.unsqueeze(0)
        T = m.shape[0]
        before_val = 1.0 if before == "regenerate_all" else 0.0
        # per-frame blend weight of the REAL mask: 0 before the gate, 1 after (+fade)
        t = torch.arange(T, dtype=torch.float32)
        fade = max(0, int(fade_frames))
        if fade > 0:
            w = ((t - start) / fade).clamp(0.0, 1.0)
        else:
            w = (t >= start).to(torch.float32)
        if bool((w >= 1.0).all()):
            return (mask,)          # whole sequence past the gate
        w = w.to(m.device).view(T, 1, 1)
        out = before_val * (1.0 - w) + m.to(torch.float32) * w
        print(f"[HGMaskTimeGate] mask gated: frames 0-{min(start, T) - 1} -> "
              f"{before} ({before_val:.0f}), kick-in at frame {start}"
              + (f", fade {fade}f" if fade else "") + f" (sequence {T} frames)")
        return (out.to(mask.dtype),)


NODE_CLASS_MAPPINGS = {"HGMaskTimeGate": HGMaskTimeGate}
NODE_DISPLAY_NAME_MAPPINGS = {"HGMaskTimeGate": "H3 Mask Time Gate (delay mask kick-in)"}
