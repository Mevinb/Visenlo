# ReactorX Swap Engine v1 — 100% Local

ReactorX is an independent **local-first** application. It does not require Stable
Diffusion WebUI or Forge. `Reactorv4` was used as reference only and is not
imported or modified.

Pipeline: detect -> dense landmarks -> align -> parse -> process references ->
aggregate identity -> swap -> CodeFormer restoration (optional) -> color match ->
occlusion recovery -> boundary blend -> identity verification -> auto-save.

## Quick start — automatic local install

All processing stays **100% on your device**. No data leaves your machine.

| OS | Command |
|---|---|
| **Linux / macOS** | `./install.sh` or `./run.sh` or `python launcher.py` |
| **Windows (PowerShell)** | `powershell -ExecutionPolicy Bypass -File install.ps1` |
| **Windows (CMD)** | double-click `install.bat` or `run.bat` |
| **Docker (optional)** | `docker compose up --build` → http://localhost:7860 |

**Option A — Git clone (recommended):**
```bash
git clone https://github.com/Mevinb/Reactor-X.git
cd ReactorX
python launcher.py              # cross-platform, handles .venv + deps + port automatically
# or
./run.sh                        # Linux/macOS
.\run.bat                       # Windows
```

**Option B — ZIP download (no git needed):**
- Download ZIP from GitHub: `Code → Download ZIP` or your website's Download button
- Extract → double-click `install.bat` (Windows) or run `./install.sh` (Linux/macOS)
- For you (maintainer) to create the ZIP: `python scripts/make_dist.py` → `dist/ReactorX-v1.zip` (2 MB, excludes `.venv`/`models/*.onnx`/`outputs`; downloader fetches deps & models on first run)

**Option C — One-line remote install (zero files needed beforehand):**
```bash
# Linux / macOS — auto-clones repo if missing:
curl -fsSL https://YOUR_WEBSITE/install.sh | bash
# Windows PowerShell — auto-clones repo if missing:
irm https://YOUR_WEBSITE/install.ps1 | iex
# Set custom repo with: REACTORX_REPO=https://github.com/Mevinb/Reactor-X.git
```
> `install.sh` / `install.ps1` detect `app.py` missing → `git clone` the repo (or ZIP via `curl`/`Invoke-WebRequest` if git unavailable) → then do the same local setup as Option A.

Open `http://127.0.0.1:7860`. The first launch creates `.venv` and installs the
Python packages locally. Use `--host 0.0.0.0` to expose it on your LAN.
`--port 7861` overrides the port; if busy the launcher picks the next free port.

### What the installer does (all locally)
1. **Fetches project files** if not present (git clone or ZIP)
2. Checks Python 3.10+ exists (guides install if missing)
3. Creates isolated `.venv`, installs `requirements.txt` (idempotent)
4. Verifies models in `models/` (see `scripts/download_models.py --check`); `buffalo_l` auto-downloads, others via helper/ZIP
5. Runs `scripts/selfcheck.py` and launches Gradio on loopback (offline-capable after install — no internet needed afterwards)

### How distribution works
You host only a light download page/redirect. The user gets the **full** `ReactorX/` folder (app.py, reactorx/, requirements.txt, scripts/, models/ placeholder, outputs/) via one of the three options above. Everything after that runs 100% on their device — no code or images ever go to your server.

## Swapping and saved outputs

- **One input, one or many targets.** Select a single target image — or any
  number of them at once — set the references/controls once, and press
  *Run identity swap*. Every selected image is swapped against the same
  references; results stream into the gallery as they finish and the report
  lists status per image. A failing image is reported and skipped without
  stopping the rest.
- **Auto-save.** Every completed swap is written to `outputs/` as
  `<date>_<NN>.png` — for example `outputs/2026-08-26_00.png`,
  `2026-08-26_01.png`, ... The counter keeps incrementing for the day, skips
  names already on disk (atomic `O_EXCL` reservation, safe for concurrent runs),
  and restarts at 00 on a new date. The filename is
  shown in the pipeline report; disable with `PipelineConfig.save_swaps=False`.

### Viewing swapped images

The **Swapped images** gallery shows thumbnails with `object-fit: contain` so the
full image is visible (no cropping). Interaction:

- **Click any thumbnail** to open the full-size preview (lightbox). The preview
  uses `contain` and is constrained to `92vw × 86vh` so tall/wide images fit
  without clipping.
- The preview header has **download**, **fullscreen**, and **close** buttons.
  Left/right halves of the preview image navigate to previous/next; `Esc` /
  arrow keys also work.
- The grid itself has **fullscreen** and **download** toggles and grows with
  content (`height="auto"`, no 420px cap). A dark `1a1a1a` letterbox keeps
  thumbnails readable.

## Gender-based face matching

The **Face matching** control has two modes:

- **Manual (target index)** (default) — swaps the target face at *Target face
  index*, exactly as before.
- **Gender match (auto)** — the pipeline reads the gender of the selected
  reference face (the one at *Reference face index*) and swaps the
  leftmost target face of that gender. The *Target face index* slider is
  hidden in this mode.

Example: a target group photo contains a man and a woman; the reference image
is a woman. With *Gender match (auto)* only the woman's face in the target is
replaced — the man is left untouched. If the target has no face matching the
reference's gender, the swap fails with a clear message listing the detected
target genders.

Gender comes from the `genderage.onnx` attribute model included in the
`buffalo_l` analysis pack (already present under
`models/insightface/models/buffalo_l/`), so no extra download is needed.
Genders are `F`/`M`; when several references are accepted the matched gender is
a majority vote across them. The selected target face's gender and age also
appear in the console log and the pipeline report (`gender-matched (F): target
face 1 of 2`).

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
curl -L -o models/codeformer.onnx     $BASE/codeformer.onnx            # restoration, CC BY-NC 4.0 (alt: models/restoration/codeformer.onnx)
curl -L -o models/inswapper_128.onnx  $BASE/inswapper_128.onnx         # swap model, research-only
# optional occlusion mask (GPL-3.0):
curl -L -o models/xseg_1.onnx https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx
```

`reswapper_256.onnx` is optional and has no canonical public host; without it
the 256px dropdown entry is unavailable. Note that `inswapper_128` and
`reswapper_256` are InsightFace research models — non-commercial use only.
Pixel-boost suffixes (`@256` etc.) are only valid for `inswapper_128`; the UI
rejects them for `reswapper_256`.

## CodeFormer restoration

Optional face-restoration stage (default off). Requires the ONNX conversion of
the full CodeFormer graph at `ReactorX/models/codeformer.onnx` (or
`models/restoration/codeformer.onnx`, ~377 MB, inputs `input [1,3,512,512]`
and `weight` float32 scalar, output the restored 512px face).

- Toggle **Enable CodeFormer restoration** in the UI.
- **CodeFormer fidelity weight** (`w`, 0..1): lower = stronger restoration,
  higher = keeps the swapped face closer to the swap output. 0.8 is a good
  default; `w` near 1 preserves identity most. The weight is now fed as
  `float32` with correct scalar/`[1]` layout detection to avoid ORT type
  errors on strict builds.
- The pipeline crops the aligned swapped face, restores it at 512px, pastes it
  back through the same feathered blend used for the plain swap, then applies
  color matching and occlusion recovery as usual.
- An automatic identity check embeds the swapped and restored aligned faces with
  the recognition model; if restoration drops reference identity noticeably it
  is skipped for that swap. The threshold is clamped for very low similarities
  (`max(0, 0.95×, -0.10)`) and compares embeddings at matched scale.

The model is loaded once and kept in memory; it runs on the CPU at roughly 3
seconds per face.

The engine requests InsightFace's 106-point landmark output when available and
falls back conservatively when a runtime only exposes five points. Skin-only
eligibility masks preserve hair, clothing, accessories, body, and background.

This app is intended for images you own or have permission to edit.

## Quality stack

- **GPU acceleration.** The venv uses `onnxruntime-gpu` (CUDA 13 wheels are
  pulled in automatically via the `cuda,cudnn` extra). Without a GPU it falls
  back to CPU transparently. If the CUDA provider is advertised but
  `libcublas`/`libcublasLt` is missing (seen as
  `Failed to load ... libonnxruntime_providers_cuda.so`), the pipeline now
  automatically falls back to `CPUExecutionProvider` and retries model load
  instead of failing permanently.
- **Pixel boost.** `inswapper_128@256` / `inswapper_128@512` / `@1024` / `@2048`
  run the 128px model on polyphase tiles of a larger aligned crop (FaceFusion
  technique), yielding 2x/4x/8x/16x sharper swaps with the same identity
  embedding. Input size is detected robustly for both `NCHW` and `NHWC` exports.
- **Face parsing (BiSeNet).** When `models/bisenet_resnet_34.onnx` is present,
  color matching and sharpening use a skin/feature interior mask derived from
  CelebAMask-HQ parsing instead of a geometric ellipse, so hair and background
  no longer dilute lighting transfer. The parser handles dynamic dims and avoids
  double-linear blurring of binary masks (now `NEAREST` with conditional resize).
- **Occlusion masking (XSeg).** When `models/xseg_1.onnx` is present, objects
  in front of the face (hair strands, glasses, hands) are detected and kept
  from the target scene. Toggle in the UI. The occluder now handles both
  `NCHW` and `NHWC` exports and inverts in crop space before warping.
- **Safer blending.** Landmark-hull face masks follow head roll; masks feather
  with a hard interior (no ghosting); occlusion recovery only touches a thin
  boundary band and blends partially. Kernel sizes are clamped to image size to
  avoid `cv2` errors on huge faces or `2048` boost.
- **Adaptive detection.** Small faces trigger one retry at higher detector
  resolution for more precise landmarks and alignment.
- **Gallery UX.** Fixed 420px crop removed; thumbnails use `contain` on a
  dark letterbox and the preview lightbox is fully zoomable/fullscreenable.

Both ONNX models above are optional; the pipeline logs and degrades gracefully
without them.

## Self-check & model helper

```bash
.venv/bin/python scripts/selfcheck.py          # 40+ checks, synthetic data only
.venv/bin/python scripts/download_models.py --check   # verify models
.venv/bin/python scripts/download_models.py           # download missing (BiSeNet/XSeg/CodeFormer)
python launcher.py --skip-check               # skip checks for faster launch
```

Runs unit checks over the engine stages, pixel-boost roundtrips, templates,
and (when present) the BiSeNet/XSeg model adapters using synthetic images.
All 40+ checks should report `ok`; missing optional models are reported as
`[skip]`.

## Docker (optional, still local)

```bash
docker compose up --build
# open http://localhost:7860
# GPU: uncomment deploy.resources in docker-compose.yml + install nvidia-container-toolkit
```
