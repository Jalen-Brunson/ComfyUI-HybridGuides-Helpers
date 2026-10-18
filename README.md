# Hybrid Guides Helpers

ComfyUI helpers for single-prompt, resumable MiniMax H3 video-to-video workflows.
These nodes work alongside the public packages; they do not replace or patch
MMH3Tools, HybridWindows, PDD Acc or ComfyUI.

## V2V Hybrid Guides workflow

[Download the single-prompt workflow](https://raw.githubusercontent.com/Jalen-Brunson/ComfyUI-HybridGuides-Helpers/main/workflows/V2V%20Hybrid%20Guides%20-%20Single%20Prompt.json),
then follow the [setup instructions](workflows/README.txt). One prompt is reused
for every chunk, including continuation runs. Gold fields are editable; blue
notes explain setup and operation. All nine references are grouped in one numbered
strip. The canvas runs horizontally, with Easy Use Set/Get nodes carrying long
connections. No project or chunk text files are required.

**Control A mode:** `true` uses Source video as a guide; `false` uses it as a
windowed reference. In guide mode, describe “the guide video” (with `body_swap:`
for the character-swap LoRA); in reference mode, use `<Video 1>` in your prompt.
Keep the mode unchanged when continuing accepted chunks.

Install the helper repository and the other public node packages listed in the
setup instructions, then restart ComfyUI. Copy
[the placeholder image](workflows/input/hybrid_guides_reference.png) into
`ComfyUI/input/`, load the workflow, and select your source video and reference
image. Start with **Accepted chunks = 0** and **New chunks = 1**.

The workflow folder includes [node sources and tested commits](workflows/NODE_SOURCES.json),
[model download links and install locations](workflows/MODEL_SOURCES.json), and
[validation results](workflows/VALIDATION.txt). All nodes have public GitHub
sources. Model weights and source media must be supplied separately.

## Install

Clone this repository into `ComfyUI/custom_nodes/ComfyUI-HybridGuides-Helpers`.

```bash
git clone https://github.com/Jalen-Brunson/ComfyUI-HybridGuides-Helpers.git ComfyUI/custom_nodes/ComfyUI-HybridGuides-Helpers
```

Install these dependencies and restart ComfyUI:

- [ComfyUI-MMH3Tools](https://github.com/ckinpdx/ComfyUI-MMH3Tools), tested at `031bb909061355d6d273dac6fe57601cf8a0197a`.
- [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite).
- FFmpeg and ffprobe on PATH.

The tested ComfyUI commit is `0f74f7fb9`, version 0.37.0. Python dependencies
come from ComfyUI and the two packages above. For Hybrid Guides sampling, use
[HybridWindows](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows) at
`cddb1ed71dbf8d8123d1845e7b9d8470061182c5` or newer for mask-aware partial overlap.
The workflow's other model and node dependencies are listed in its
[setup instructions](workflows/README.txt).

## Nodes

All node IDs start with `HG`, so existing local nodes can remain installed.

| Nodes | Purpose |
|---|---|
| HG Cached Reference Conditioning | Reuse one prompt for every chunk, cache reference encodes, and advance reference-video windows |
| HG Cached Source Encode | Cache source-video VAE clips on disk and read only missing source ranges |
| HG Reference Tags | Validate and renumber picture, video and audio reference tags after optional slots are disabled |
| HG Resume Plan / Load Resume Run | Plan fresh or resumed spans and find the corresponding saved master/video pair |
| HG Save Latent Master / Continue Latent Master / Merge Latent Masters | Save native warmup metadata, protect accepted output and assemble the full latent timeline |
| HG Create Resume Video / Save Video and Resume Pair | Append newly generated video and audio, retain compatible prefix packets and record the completed pair |
| HG Load Audio Range | Read the source audio matching a frame span |
| HG Optional Image / Optional Input | Skip disabled upstream branches |
| HG Mask Threshold / Mask Start Time | Prepare optional inpainting masks |

## Use

Write the complete prompt once. `HGReferenceConditioning` treats the entire
string as one prompt, including any literal pipe characters. With windowed
reference video enabled, connect the same chunk length and overlap used by the
sampler. Reference encodes use the public MMH3Tools implementation.

Start with accepted chunks set to zero. To continue, keep the run name, version,
source, model settings, dimensions, FPS and chunk settings, then select the
number of completed chunks to keep. New chunks set to zero processes the
remaining source. Keep the saved video, latent master and resume records together.
No project or chunk text files are required.

Reference caches live under ComfyUI's temporary directory and are expendable.
Source caches default to `output/latents/source_cache`. Resume records live in
`output/latents/h3_resume_runs`; these are metadata, not prompt files. Graded-prefix
selection, color correction and experimental samplers are not included.

## Tests

Run with ComfyUI's Python environment. When the repository is outside the
`custom_nodes` folder, set `COMFYUI_PATH`, `MMH3TOOLS_PATH` and `VHS_PATH`:

```bash
python -m unittest discover -s tests
```

Tests cover cache reuse and invalidation, prompt/window behavior, resume pair
selection, collision-free output names, video packet reuse, frame boundaries
and audio alignment. They use small CPU tensors and synthetic media, without
loading H3 model weights.

## Source

The prompt, latent and source-cache helpers were extracted from local additions
to Jalen-Brunson's `vlm_video_prompt`, `wan_chunk_io` and mask utilities. Resume
and media helpers were extracted from local MMH3Tools additions. Reference
conditioning calls the public [MMH3Tools](https://github.com/ckinpdx/ComfyUI-MMH3Tools)
implementation by ckinpdx and adds per-reference caching. See `PROVENANCE.json`
for the original modules.
