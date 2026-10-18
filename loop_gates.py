class HGImageGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "gate"
    CATEGORY = "Hybrid Guides"

    def check_lazy_status(self, enabled, image=None):
        # Only pull (and thereby execute) the upstream branch when enabled.
        if enabled and image is None:
            return ["image"]
        return []

    def gate(self, enabled, image=None):
        if enabled and image is not None:
            return (image,)
        return (None,)

class HGAnyGate:
    """HGImageGate for arbitrary types: pass `value` through when `enabled`,
    emit None otherwise. The value input is lazy, so the whole upstream
    branch (VAE encodes, mask conversion, ...) is skipped while the gate is
    off — this is how optional sub-flows (e.g. v4 v2v inpaint) toggle without
    paying for their disabled chain. Downstream OPTIONAL inputs receive None,
    which the MMH3 sampler / HGContinueMaster treat as 'not provided'.
    `enabled` is INT 0/1 (not BOOLEAN) so one INT toggle can drive gates AND
    easy index-switch `index` inputs from the same wire."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("INT", {"default": 1, "min": 0, "max": 1,
                                    "tooltip": "0 = emit None (upstream branch skipped), 1 = pass value through."}),
            },
            "optional": {
                "value": ("*", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "gate"
    CATEGORY = "Hybrid Guides"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True  # wildcard: accept any input/output pairing

    def check_lazy_status(self, enabled, value=None):
        if enabled and value is None:
            return ["value"]
        return []

    def gate(self, enabled, value=None):
        if enabled and value is not None:
            return (value,)
        return (None,)

class HGMaskBinarize:
    """Snap a mask to pure 0/1 at a threshold, optionally inverted.

    The MMH3 sampler reduces pixel masks with max-pooling but never
    binarizes VALUES — any gray that reaches it (limited-range mask video,
    codec halos, a loader rescale's edge band) becomes a per-token
    fractional timestep (rows_t = 1 - m*sigma): partially-pinned regen
    regions and re-noised 'kept' rows. One threshold here makes the whole
    upstream chain immune. Invert swaps polarity (white becomes the KEPT
    region) for freeze-the-mask flows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Values above this become 1.0 (regenerate), the rest 0.0 (keep)."}),
                "invert": ("BOOLEAN", {"default": False,
                                       "tooltip": "Swap polarity after thresholding: white areas become the KEPT region."}),
            },
            "optional": {
                "white_level": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Value the REGENERATE region gets after snapping. 1.0 = full replacement (default). 0.90-0.95 = the H3 'band': the model faintly sees the source under the mask, so its facial performance and timing bleed through as guidance while the region still regenerates - a deliberate likeness<->performance dial. Below ~0.9 the model just recreates the original (community-measured cliff); values there are refused down to 0.9 unless set to exactly 0 for testing."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "binarize"
    CATEGORY = "Hybrid Guides"

    def binarize(self, mask, threshold, invert, white_level=1.0):
        out = (mask > threshold).to(mask.dtype)
        if invert:
            out = 1.0 - out
        if white_level < 1.0:
            if 0.0 < white_level < 0.9:
                print(f"[HGMaskBinarize] white_level {white_level} is below the usable "
                      "H3 band (0.90-0.95) - the model will mostly recreate the "
                      "source there; clamping to 0.9")
                white_level = 0.9
            out = out * white_level
        return (out,)
