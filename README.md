# ReactorX Swap Engine v1

ReactorX is an independent local application. It does not require Stable
Diffusion WebUI or Forge. `Reactorv4` was used as reference only and is not
imported or modified.

Pipeline: detect -> dense landmarks -> align -> parse -> process references ->
aggregate identity -> swap -> CodeFormer restoration (optional) -> color match ->
occlusion recovery -> boundary blend -> identity verification.

## Start

```bash
cd ReactorX
./run.sh
```

Open `http://127.0.0.1:7860`. The first launch creates `.venv` and installs the
Python packages. Use `./run.sh --host 0.0.0.0` to expose it on your LAN.

## Models

The local `buffalo_l` analysis pack has already been copied into
`ReactorX/models/insightface/models/buffalo_l/` from the existing Desktop model
cache, so those files will not be downloaded again. The five copied files are
`1k3d68.onnx`, `2d106det.onnx`, `det_10g.onnx`, `genderage.onnx`, and
`w600k_r50.onnx`.

The existing Forge installation contained both models. They have been copied
and validated at `ReactorX/models/inswapper_128.onnx` and
`ReactorX/models/reswapper_256.onnx`. ReactorX uses the standard InsightFace
adapter for the 128px model and a dedicated two-input ONNX adapter for the
256px model, because InsightFace 0.7.3 otherwise misclassifies the latter as
an ArcFace recognition model.
Set `REACTORX_MODELS=/another/path` to use another model directory.

### Getting the models

Model weights are **not included in this repository** (size and license
restrictions). The `buffalo_l` analysis pack downloads automatically into
`models/insightface/` on first launch. The remaining files can be fetched from
their upstream hosts:

```bash
BASE=https://huggingface.co/facefusion/models-3.0.0/resolve/main
curl -L -o models/bisenet_resnet_34.onnx $BASE/bisenet_resnet_34.onnx   # face parsing, MIT
curl -L -o models/codeformer.onnx     $BASE/codeformer.onnx            # restoration, CC BY-NC 4.0
curl -L -o models/inswapper_128.onnx  $BASE/inswapper_128.onnx         # swap model, research-only
# optional occlusion mask (GPL-3.0):
curl -L -o models/xseg_1.onnx https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx
```

`reswapper_256.onnx` is optional and has no canonical public host; without it
the 256px dropdown entry is unavailable. Note that `inswapper_128` and
`reswapper_256` are InsightFace research models — non-commercial use only.

## CodeFormer restoration

Optional face-restoration stage (default off). Requires the ONNX conversion of
the full CodeFormer graph at `ReactorX/models/codeformer.onnx` (~377 MB, inputs
`input [1,3,512,512]` and `weight`, output the restored 512px face).

- Toggle **Enable CodeFormer restoration** in the UI.
- **CodeFormer fidelity weight** (`w`, 0..1): lower = stronger restoration,
  higher = keeps the swapped face closer to the swap output. 0.8 is a good
  default; `w` near 1 preserves identity most.
- The pipeline crops the aligned swapped face, restores it at 512px, pastes it
  back through the same feathered blend used for the plain swap, then applies
  color matching and occlusion recovery as usual.
- An automatic identity check embeds the swapped and restored aligned faces with
  the recognition model; if restoration drops reference identity noticeably it
  is skipped for that swap.

The model is loaded once and kept in memory; it runs on the CPU at roughly 3
seconds per face.

The engine requests InsightFace's 106-point landmark output when available and
falls back conservatively when a runtime only exposes five points. Skin-only
eligibility masks preserve hair, clothing, accessories, body, and background.

This app is intended for images you own or have permission to edit.

## Quality stack

- **GPU acceleration.** The venv uses `onnxruntime-gpu` (CUDA 13 wheels are
  pulled in automatically via the `cuda,cudnn` extra). Without a GPU it falls
  back to CPU transparently.
- **Pixel boost.** `inswapper_128@256` / `inswapper_128@512` run the 128px
  model on polyphase tiles of a larger aligned crop (FaceFusion technique),
  yielding 2x/4x sharper swaps with the same identity embedding.
- **Face parsing (BiSeNet).** When `models/bisenet_resnet_34.onnx` is present,
  color matching and sharpening use a skin/feature interior mask derived from
  CelebAMask-HQ parsing instead of a geometric ellipse, so hair and background
  no longer dilute lighting transfer.
- **Occlusion masking (XSeg).** When `models/xseg_1.onnx` is present, objects
  in front of the face (hair strands, glasses, hands) are detected and kept
  from the target scene. Toggle in the UI.
- **Safer blending.** Landmark-hull face masks follow head roll; masks feather
  with a hard interior (no ghosting); occlusion recovery only touches a thin
  boundary band and blends partially.
- **Adaptive detection.** Small faces trigger one retry at higher detector
  resolution for more precise landmarks and alignment.

Both ONNX models above are optional; the pipeline logs and degrades gracefully
without them.

## Self-check

```bash
.venv/bin/python scripts/selfcheck.py
```

Runs unit checks over the engine stages, pixel-boost roundtrips, templates,
and (when present) the BiSeNet/XSeg model adapters using synthetic images.
