V2V HYBRID GUIDES — SINGLE PROMPT
Public workflow • 2026-10-18

QUICK START
1. Use current ComfyUI with MiniMax H3 and subgraph support. Prepared against
   ComfyUI 0.37.0 and frontend 1.53.6. Install FFmpeg (including ffprobe).
2. Clone the helper repository into ComfyUI/custom_nodes/ComfyUI-HybridGuides-Helpers:
   git clone https://github.com/Jalen-Brunson/ComfyUI-HybridGuides-Helpers.git ComfyUI/custom_nodes/ComfyUI-HybridGuides-Helpers
   Install the eleven other public packs listed below into ComfyUI/custom_nodes/.
   No modified MMH3Tools, PDD or ComfyUI files are required.
   Helper node IDs start with HG to avoid local-node collisions.
3. Install pack requirements with ComfyUI's Python environment. Use the tested
   commits in NODE_SOURCES.json, especially the RMBG version below. Restart ComfyUI.
4. Download the models below into the specified folders. Model weights are not included.
   BEN2 background-removal weights are obtained by the RMBG pack on first use.
5. From this repository’s workflows/ folder, copy input/hybrid_guides_reference.png
   into ComfyUI/input/. This neutral placeholder
   keeps unused image slots valid; replace Reference 1 with your actual character image.
6. Drag “V2V Hybrid Guides - Single Prompt.json” into ComfyUI.
7. Edit the GOLD fields. Source video takes input/your_video.mp4 (relative to the
   ComfyUI directory) or an absolute path. This one field supplies Control A and
   source audio/inpaint input. Write ONE prompt; it is reused in every chunk.
8. Choose a unique run name. Start with Accepted chunks = 0, New chunks = 1.
   Upload Reference 1, select the installed model filenames, then Queue.

PROMPTS
Guide mode is ON. Example:
  body_swap: Replace the main subject in the guide video with the character in
  <Picture 1>. Preserve the character's face, hair, clothing, and proportions.
  Follow the motion, timing, camera, and composition of the guide video.
  Keep the background consistent and the movement natural.

Write the common instruction once. New and resumed chunks use the same prompt.
The entire text is one prompt, including any literal | characters. There is no
project TXT or chunks TXT input, and no prompt file to create.
Reference 1 is on by default. Reference 2–9 are off: enable a slot before naming its
<Picture N> tag. Tags are renumbered automatically after disabled references drop out.
To use Control A as a regular reference, turn GUIDE off and refer to <Video 1>.
Choose a prompt/LoRA suited to reference mode. Control A's enable gate must remain on.

CONTINUATION
Keep the run name, version, source, seed, dimensions, FPS, chunk length and overlap.
Set Accepted chunks to the number of completed chunks to keep, then New chunks to
how many to add. New chunks = 0 means all remaining chunks in the source.
The resume loader selects the matching completed video and latent master. Its master
selector can choose an older run. Use a new run name for another video or settings.
This plain run name only labels saved outputs; it is not a project file.
The flow still writes its own video, latent master and resume-index data automatically.
Keep those outputs together to resume. Do not edit the index or rename saved masters.
Video: output/video/HybridGuides/<run name>/video_<version>_....mp4
Masters: output/latents/ (run-scoped names); resume records: output/latents/h3_resume_runs/.

LAYOUT
Blue notes explain setup and operation. Gold nodes are editable inputs/settings.
Gray nodes process data. Sections read top to bottom: instructions, main controls,
references, models, extra references, optional inputs, planning, sampling and output.
Open a processing subgraph to inspect its internals. The guide video reaches both
sampling resolutions through H3HybridWindows' integrated guide inputs.

BASELINE / CHANGES
Preserves Ref2VA, character-swap v1.1 at 1.0, sigma shift 12/3, Sol attention settings,
PDD 8-NFE weights with seven warmup steps, 1-MP clean latent upscale, and DMAD at 0.8
for two joint finish steps. Default dimensions 864x480; 24 FPS; 10-second chunk
request snaps to 243 frames; overlap 39. Native warmup context is saved for resume.

The run now starts fresh with one chunk. Private paths/images/prompts and old text-file
readers are removed. The inactive experimental patches, diagnostics, scene reporter,
finish-grade launcher and custom live-preview override have been removed. Core preview
can be used. Hair masks, inpainting, RefMods and nine image slots remain. Unconnected extra
control-video and likeness branches were removed.
The sample PNG is not a character reference. No original media or private prompts ship.

DEPENDENCIES — PUBLIC PACKS
  ComfyUI-HybridWindows
  https://github.com/Jalen-Brunson/ComfyUI-HybridWindows
  ComfyUI-MMH3Tools
  https://github.com/ckinpdx/ComfyUI-MMH3Tools
  minimax_h3_pdd_acc
  https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc
  Comfyui_Minimax_h3_latent_Upscaler
  https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
  comfy-plasma
  https://github.com/Jordach/comfy-plasma
  ComfyUI-MiniMaxH3Mod
  https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod
  ComfyUI-RMBG
  https://github.com/1038lab/ComfyUI-RMBG
  MaskVidExperiments
  https://github.com/drozbay/MaskVidExperiments
  comfyui-custom-scripts
  https://github.com/pythongosssss/ComfyUI-Custom-Scripts
  comfyui-easy-use
  https://github.com/yolain/ComfyUI-Easy-Use
  comfyui-videohelpersuite
  https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

RMBG VERSION (important)
  Use commit d7402513f23f58db7d56754b02a4f51a148b4941. Newer RMBG versions add an
  extra widget that shifts saved values in older workflows. After cloning:
    git -C ComfyUI/custom_nodes/ComfyUI-RMBG checkout d7402513f23f58db7d56754b02a4f51a148b4941
  Then install its requirements with ComfyUI's Python. All other tested public
  commits are listed in NODE_SOURCES.json.

HELPER REPOSITORY
  ComfyUI-HybridGuides-Helpers
  https://github.com/Jalen-Brunson/ComfyUI-HybridGuides-Helpers

This repository contains the prompt, source/reference caches,
resume planning, saved master/video pairing and optional input/mask utilities.
Public MMH3Tools supplies reference encoding, RefMods and the standard latent tools.
Public HybridWindows includes the mask-aware overlap fix. PDD uses its ordinary
8-step partition. Experimental samplers, color corrections, graded-prefix selection,
static-video exceptions and off-envelope PDD partitions are excluded.

NODE_SOURCES.json maps every node to its public repository and tested version.
The helper code lives at the repository root; this workflow and its assets live in workflows/.
Dependencies are provided by ComfyUI and the listed public packs.

MODELS — EXACT FILES AND INSTALL LOCATIONS

  ComfyUI/models/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors
  https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors

  ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
  https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors

  ComfyUI/models/vae/minimax_h3_video_vae_int8_convrot.safetensors
  https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_int8_convrot.safetensors

  ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors
  https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors

  ComfyUI/models/loras/minimax/minimax_h3_character_swap_v1.1_ref2va_fro.safetensors
  https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/loras/minimax_h3_character_swap_v1.1_ref2va_fro.safetensors

  ComfyUI/models/loras/minimax/minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors
  https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors

  ComfyUI/models/pdd_acc/minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors
  https://huggingface.co/aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI/blob/main/minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors

  ComfyUI/models/latent_upscale_models/h3_upscaler_lms_v0.1_fp32.safetensors
  https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/latent_upscaler/h3_upscaler_lms_v0.1_fp32.safetensors

The selected INT8 ConvRot preset needs a compatible recent PyTorch/CUDA/Comfy Kitchen
installation. Other H3 model formats may require changing the loader selection and
performance settings. This is a large H3 workflow; hardware capacity was not profiled
for smaller GPUs. Start with one chunk before asking for the full source.

VALIDATION
See VALIDATION.txt for the checks performed on this release and their limits.
Share the helper repository link for the workflow, setup notes and helper source:
https://github.com/Jalen-Brunson/ComfyUI-HybridGuides-Helpers
Every node can be sourced from the public repositories in NODE_SOURCES.json.
