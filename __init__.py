"""Prompt, cache and continuation helpers for V2V Hybrid Guides."""
from .latent import HGSaveAVLatent, HGContinueMaster, HGMergeMasters
from .tags import HGTagRenumber
from .source_encode_cache import HGCachedSourceEncode
from .mask_time_gate import HGMaskTimeGate
from .loop_gates import HGImageGate, HGAnyGate, HGMaskBinarize
from .resume import HGLoadResumeRun, HGSaveResumeVideo, HGResumePlan
from .media import HGResumeVideo, HGAudioRange
from .reference import HGReferenceConditioning

NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    HGSaveAVLatent, HGContinueMaster, HGMergeMasters, HGTagRenumber,
    HGCachedSourceEncode, HGMaskTimeGate, HGImageGate, HGAnyGate, HGMaskBinarize,
    HGLoadResumeRun, HGSaveResumeVideo, HGResumePlan, HGResumeVideo, HGAudioRange,
    HGReferenceConditioning,
)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HGSaveAVLatent": "HG Save Latent Master",
    "HGContinueMaster": "HG Continue Latent Master",
    "HGMergeMasters": "HG Merge Latent Masters",
    "HGTagRenumber": "HG Reference Tags",
    "HGCachedSourceEncode": "HG Cached Source Encode",
    "HGMaskTimeGate": "HG Mask Start Time",
    "HGImageGate": "HG Optional Image",
    "HGAnyGate": "HG Optional Input",
    "HGMaskBinarize": "HG Mask Threshold",
    "HGLoadResumeRun": "HG Load Resume Run",
    "HGSaveResumeVideo": "HG Save Video and Resume Pair",
    "HGResumePlan": "HG Resume Plan",
    "HGResumeVideo": "HG Create Resume Video",
    "HGAudioRange": "HG Load Audio Range",
    "HGReferenceConditioning": "HG Cached Reference Conditioning",
}
