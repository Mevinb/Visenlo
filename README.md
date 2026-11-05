# Visenlo · local face studio

Visenlo is an independent **local-first** face-swap application. It does not
require Stable Diffusion WebUI or Forge. `Reactorv4` was used as reference
only and is not imported or modified.

> **Renamed:** this project was previously called **ReactorX**. Everything is
> now **Visenlo** — folder `Visenlo/`, package `visenlo/`, env
> `VISENLO_MODELS` / `VISENLO_REPO` / `VISENLO_ZIP`, class `VisenloPipeline`,
> Docker service `visenlo`, dist `Visenlo-v1.zip`. Old `REACTORX_*` env vars
> and `ReactorXPipeline` still work as aliases so existing setups don't break.

Pipeline (per target face): detect → 106-point landmarks → refined 5-point
alignment → parse → **TargetContext** (masks + ROI, built once) → swap
candidates (**Pixel Boost**, capped at the face's real resolution, de-aliased)
→ crop-space ArcFace + sharpness scoring → guarded restoration (CodeFormer /
GFPGAN / GPEN / RestoreFormer++, blended) → region-mask multi-band paste →
colour match / occlusion / mouth / eyes → auto-save.

## What changed in v2 (same `inswapper_128` model)

| Area | Before | Now |
|---|---|---|
| Best likeness cost | 12 full-frame detections + 6 full-frame post-passes per image | 0 detections after the first; scoring embeds the aligned crop at the known landmarks; masks built once; post-processing runs on the face ROI only. **2.9 s → 0.8 s** per image on an RTX 4050 |
| Best likeness choice | ties went to the *lowest* boost (blurriest) | ties go to the **sharpest** candidate; boosts that would upsample the source face are skipped |
| Paste-back | square erode+feather box, bilinear | parsed **face-region mask** (hair/background/box edge excluded) + **Laplacian multi-band** blend, Lanczos |
| Alignment | SCRFD's raw 5 points | eye centroids + mouth corners from the 106-point contours (`refine_landmarks`) |
| GPU | fp32 only, 4 threads hard cap | `inswapper_128_fp16.onnx` twin on CUDA (same weights, ~2x faster tiles), `VISENLO_THREADS`, opt-in TensorRT (`VISENLO_TRT=1`) |
| Restoration | CodeFormer accept/reject | CodeFormer, GFPGAN 1.4, GPEN BFR 512, RestoreFormer++ with a **blend** slider, all under the identity guard |
| Faces | one face per run, index slider | **face picker strip**, multi-person **face mapping** (A–D), gender match, *looks-like-anchor* (similarity) mode, *all faces* |
| People | upload 1–4 photos each time | **identity bank** — save a person once, pick from a dropdown (embedding only, no photos stored) |
| Video | — | **video tab**: identity tracking, temporal smoothing, ffmpeg re-encode with audio |
| UX | — | before/after slider, mask debug view, Cancel, settings presets, `--auth` |
| Automation | — | `python -m visenlo swap|video|faces|identity …` headless CLI |
| Tests | ad-hoc scripts | `pytest tests/` (32 tests) + `scripts/selfcheck.py` |

Measured but **not** adopted: CUDA graphs and IO binding gave no speed-up for
this graph (it is compute-bound at ~52 ms/tile fp32, ~27 ms fp16 on a 4050).

## Current studio workflow

The studio has four sections: **Choose your scene** (Images or Video tab),
**Add people**, **Make it yours**, and **Your results**. Add target images (a
face strip shows every detected face, numbered left → right) or a video, then
reference photos for **Person A** — or pick a saved identity. People B–D are
optional and only needed for mapping several faces. Choose *which face gets
replaced*: single index, gender match, all faces, "looks like anchor photo", or
map each detected face to a person. Fine-tune holds blending, restoration,
masks and presets. Results stream into the gallery; click a result to open the
before/after slider.

**First-run setup happens inside the local app.** Open **Model setup**, review the
linked model terms, tick the acknowledgement and click **Install required models**.
The setup report streams progress, keeps existing weights, and refreshes readiness
when finished. Optional finishing models are a separate checkbox. No download
starts from opening the app or checking readiness. You can retry failed transfers.

Advanced terminal setup remains available after reviewing model licenses:

```bash
.venv/bin/python scripts/download_models.py --download
# Add optional finishing models:
.venv/bin/python scripts/download_models.py --download --all
```

Windows uses `.venv\Scripts\python.exe`. Private images, local environments and
weights are excluded from source ZIPs.

## Quick start — automatic local install

All processing stays **100% on your device**. No data leaves your machine.

| OS | Command |
|---|---|
| **Linux / macOS** | `./run.sh` |
| **Windows (PowerShell)** | `.\run.bat` |
| **Windows (CMD)** | double-click `run.bat` |
| **Docker (optional)** | `docker compose up --build` → http://localhost:7860 |

**Option A — Git clone (recommended):**
```bash
git clone https://github.com/Mevinb/Visenlo.git
cd Visenlo
./run.sh                        # Linux/macOS: sets up and launches the app
```

On Windows, use `.\run.bat` after cloning instead of `./run.sh`.

**Option B — ZIP download (no git needed):**
- Download ZIP from GitHub: `Code → Download ZIP` or your website's Download button
- Extract → open the `Visenlo/` folder containing the run scripts → double-click `run.bat` (Windows) or run `./run.sh` (Linux/macOS)
- For you (maintainer) to create the ZIP: `python scripts/make_dist.py` → `dist/Visenlo-v1.zip` (source only; excludes environments, models, private inputs, and outputs; models require explicit setup)

**Advanced maintainer option — hosted bootstrap scripts:**

The promotional site serves the source ZIP. It does not expose a hosted shell
installer. The following placeholders are examples only for maintainers who
choose to separately host the bootstrap scripts; use the ZIP/run-script flow above.
```bash
# Linux / macOS — auto-clones repo if missing:
curl -fsSL https://YOUR_WEBSITE/install.sh | bash
# Windows PowerShell — auto-clones repo if missing:
irm https://YOUR_WEBSITE/install.ps1 | iex
# Set custom repo with: VISENLO_REPO=https://github.com/Mevinb/Visenlo.git
# Legacy REACTORX_REPO / REACTORX_ZIP are still honored as fallback.
```
> `install.sh` / `install.ps1` detect `app.py` missing → `git clone` the repo (or ZIP via `curl`/`Invoke-WebRequest` if git unavailable) → then do the same local setup as Option A.

`run.sh` and `run.bat` call the Python bootstrap internally; users can start
the app directly with the appropriate run script. If a ZIP extractor removes
Unix execute permissions, run `chmod +x run.sh` once before `./run.sh`.

The local studio opens in your browser automatically. If it does not, use the
URL printed in the terminal (normally `http://127.0.0.1:7860`). Pass `--no-browser`
to suppress browser opening. The first launch creates `.venv` and installs the
Python packages locally. Later launches reuse working dependencies without reinstalling them. Use `--host 0.0.0.0` to expose it on your LAN.
`--port 7861` overrides the port; if busy the launcher picks the next free port.

### What the installer does (all locally)
1. **Fetches project files** when using the hosted bootstrap installer (git clone or ZIP)
2. Checks Python 3.10+ exists (guides install if missing)
3. Creates isolated `.venv`, installs `requirements.txt` (working installs reused, CPU/GPU auto-detected)
4. **Model checker** — runs `scripts/download_models.py --check` (no auto-download, licensing) → if missing, shows **Model Setup Guide**
5. Opens the local studio on loopback. Extended diagnostics run only with `./run.sh --self-check`; normal startup does not run the full model test suite.

### First launch flow (licensing-safe)
```
First launch
    ↓
Model checker (scripts/download_models.py --check)
    ↓
Missing model?
    ↓
Open Model setup → review model terms → acknowledge → Install required models
    ↓
Download progress → automatic readiness refresh
    ↓
[✓] buffalo_l   [✓] selected swap model   [optional] BiSeNet / XSeg / CodeFormer
                          ↓
                [Run identity swap]
```

### How distribution works
You host only a light download page/redirect. The user gets the **full** `Visenlo/` folder (app.py, visenlo/, requirements.txt, scripts/, models/ placeholder, outputs/) via one of the three options above. Everything after that runs 100% on their device — no code or images ever go to your server. The app's top **Model setup** accordion shows live checkmarks and disables *Run identity swap* until required models are present.

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

## People, face mapping and the identity bank

- **Face strip.** After adding targets, the faces of the first image appear as
  numbered thumbnails (`Face 0 · M · ~34`). Indices are left → right.
- **Which face gets replaced**
  - *Single face (index)* — the classic slider.
  - *Gender match (auto)* — leftmost face with Person A's gender.
  - *All faces → Person A* — every detected face.
  - *Looks like anchor photo* — upload a photo of the person currently in the
    scene; the most similar target face (ArcFace ≥ 0.25) is replaced. This is
    also how video tracking works.
  - *Map faces manually* — one dropdown per detected face: Keep / Person A–D.
    Faces are swapped in order into the same frame; the per-face report lists
    boost, similarity and restoration for each.
- **Identity bank.** *Save identity* stores Person A's aggregated embedding (plus
  gender/age and a thumbnail) under `models/identities/<name>.npz`. Pick it from
  "saved identity" next time — no photos needed. Image-conditioned research
  backends (BlendSwap, UniFace, FaceDancer, MobileFaceSwap) still need photos.

## Video

Upload a clip in the **Video** tab, set people and the assignment mode, press
*Run video swap*. Index/gender picks are resolved on the first frame with faces
and then tracked by identity similarity (the anchor drifts slowly so pose and
lighting changes stay locked). *Temporal smoothing* steadies the alignment;
*Swap every Nth frame* gives quick previews. Video always uses Manual quality
(Best likeness would flicker). The result is re-encoded with ffmpeg (libx264,
CRF 18) and keeps the source audio; without ffmpeg a silent `mp4v` file is kept.

## Command line

```bash
.venv/bin/python -m visenlo faces photo.jpg
.venv/bin/python -m visenlo swap --ref maya1.jpg --ref maya2.jpg --target group.jpg --match all --out out/
.venv/bin/python -m visenlo swap --identity maya --target a.jpg --target b.jpg --model inswapper_128.onnx@512 --manual
.venv/bin/python -m visenlo video --identity maya --input clip.mp4 --face 1 --restorer gfpgan_1.4
.venv/bin/python -m visenlo identity save maya --ref maya1.jpg --ref maya2.jpg
.venv/bin/python -m visenlo identity list
```

Flags mirror the UI: `--restorer`, `--restore-blend`, `--mask box|region`,
`--blend alpha|multiband`, `--mouth`, `--keep-eyes`, `--precision`, `--cpu`.

## Settings reference (`PipelineConfig`)

| Setting | Default | Meaning |
|---|---|---|
| `quality_mode` | `best_likeness` | compare x1/x2/x4 (up to the face's native resolution) vs. run the selection once |
| `likeness_tolerance` | 0.01 | ArcFace scores closer than this are a tie → sharpest wins |
| `boost_slack` / `boost_dealias` | 2.0 / off | max upsample allowed before extra tiles are skipped; optional median softening of the tile grid |
| `mask_mode` / `blend_mode` | `region` / `multiband` | parsed-face blend mask; Laplacian pyramid blend |
| `refine_landmarks` | on | 106-point eye centroids / mouth corners for alignment |
| `restorer` / `restoration_blend` / `codeformer_weight` | `codeformer` / 0.7 / 0.9 | restorer key, mix-in amount, CodeFormer fidelity |
| `mouth_mode` / `eye_mode` | `swap_lips_keep_teeth` / `swap` | keep target teeth / mouth / eyes |
| `inswapper_precision` | `auto` | fp16 twin on CUDA when present |
| `similar_threshold` | 0.25 | anchor / video tracking similarity gate |
| `debug_masks` | off | return face/interior/occlusion/mouth/eye/blend masks |

Environment: `VISENLO_MODELS`, `VISENLO_DEVICE=cpu`, `VISENLO_THREADS=N`,
`VISENLO_TRT=1`, `VISENLO_EXPERT=1` (shows `@2048`), `VISENLO_CONFIG_DIR`.

## Pixel Boost — explicit guide (read this)

**Pixel Boost is how Visenlo gets sharp swaps out of the 128px
`inswapper_128` model without changing identity.** The same identity embedding
is reused — only the aligned crop resolution changes.

### The 5 choices in the UI

| Dropdown entry | Aligned crop | Boost factor | Tiles run | Sharpness | Relative cost |
|---|---|---|---|---|---|
| `inswapper_128.onnx` | 128px | x1 (native) | 1 | 1x | 1x |
| `inswapper_128.onnx@256` | 256px | x2 | 4 | ~2x | ~4x |
| `inswapper_128.onnx@512` | 512px | x4 | 16 | ~4x | ~16x |
| `inswapper_128.onnx@1024` | 1024px | x8 | 64 | ~8x | ~64x |
| `inswapper_128.onnx@2048` | 2048px | x16 | 256 | ~16x | ~256x |
| `reswapper_256.onnx` | 256px native | — (no boost) | 1 | — | 1x |

Suffix = **desired aligned resolution**. Internally `parse_swapper_spec()`
converts it to a scale factor relative to the model's native 128px:
`256→x2`, `512→x4`, `1024→x8`, `2048→x16`. Short forms `@2/@4/@8/@16` and
legacy `inswapper_128@256` (auto-gains `.onnx`) resolve identically.

### How it works under the hood (`visenlo/boost.py`)

1. `face_align.norm_crop2(image, kps, 128*factor)` aligns a **larger** crop
   (`big = 128 × factor`).
2. `implode_tiles()` splits it into `factor²` whole-face views, each
   downsampled by `factor` at sub-pixel phase offsets (polyphase
   decomposition, FaceFusion technique). Every tile keeps the model's expected
   128px receptive field, so there are **no seams**.
3. Tiles run **sequentially** through `inswapper_128.onnx` (the graph declares
   fixed batch 1) with the same source identity embedding.
4. `explode_tiles()` interleaves the outputs back to the full `big × big` face.
5. `_paste_back()` warps it to the frame with **Lanczos** (not bilinear) and the
   same erode + feather mask logic as the native swapper.

Input size is detected robustly for both `NCHW` and `NHWC` exports, and mask
kernel sizes are clamped so `@2048` never crashes `cv2` on huge faces.

### Explicit choice vs Best likeness (automatic)

- **Explicit `@256/@512/@1024/@2048` always wins — in both quality modes.**
  If you pick `@512`, Visenlo runs exactly x4, no comparison, no override.
  The report says `requested pixel-boost 512px (x4)`.
- **Plain `inswapper_128.onnx` + Best likeness (default):** Visenlo runs
  **native, 256px, and 512px** candidates (`candidate_boosts_for() → (1, 2, 4)`),
  **skipping boosts that would upsample the source face more than 2x**
  (`max_useful_boost()` — a 250px face gets x1/x2, a 400px face x1/x2/x4), each in **two
  finishes** (`default` + `identity-preserving` with color/sharpen off). It
  embeds each composited face at the target landmarks with ArcFace (no
  re-detection), treats scores within 0.01 as a tie and keeps the **sharpest**
  candidate (`select_likeness_candidate()`), reporting
  `best-likeness x2 default (compared 128/256/512)`. “Reference similarity” is
  a comparison value, not a guarantee the image looks convincing.
- **Tile grid (optional softening).** Polyphase tiles can leave a faint
  period-`factor` grid. `boost_dealias` (off by default) applies a median over
  that footprint — measured on real faces it also removes ~75% of the
  sharpness pixel boost adds, so only turn it on if the grid bothers you.
- **Plain `inswapper_128.onnx` + Manual:** runs **native x1 only**.
- **Pixel Boost is rejected for `reswapper_256.onnx`.** The UI suffixes are only
  valid for `inswapper_128`; the pipeline raises
  `Pixel-boost (@256/@512/…) is not supported for reswapper_256.onnx`.

### When to use what

- **Default (Best likeness, plain 128px):** leave it — best quality/effort
  trade-off for most faces.
- **`@256`:** visibly crisper eyes/teeth on close-ups, still fast. Good first
  manual step up.
- **`@512`:** large prints / 1K+ faces. Noticeably sharper, ~16 tile passes.
- **`@1024` / `@2048`:** forensic crops / huge posters only. 64–256 tile
  passes — use a GPU, expect seconds-to-minutes on CPU. Gains beyond `@512`
  are subtle unless the target face itself is 800px+.

## Gender-based face matching

The **Face matching** control under **Fine-tune** has two modes:

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

Gender matching uses a fallible classifier estimate and may select unexpectedly; manual face selection is the default. Gender comes from the `genderage.onnx` attribute model included in the
`buffalo_l` analysis pack (installed under
`models/insightface/models/buffalo_l/`), so no extra download is needed.
Genders are `F`/`M`; when several references are accepted the matched gender is
a majority vote across them. The selected target face's gender and age also
appear in the console log and the pipeline report (`gender-matched (F): target
face 1 of 2`).

## Models

Keep the `buffalo_l` analysis pack in
`Visenlo/models/insightface/models/buffalo_l/`. Existing files are reused. The five files are
`1k3d68.onnx`, `2d106det.onnx`, `det_10g.onnx`, `genderage.onnx`, and
`w600k_r50.onnx`.

The default swap weights belong at `Visenlo/models/inswapper_128.onnx`.
An optional ReSwapper installation belongs at `Visenlo/models/reswapper_256.onnx`. Visenlo uses the standard InsightFace
adapter for the 128px model and a dedicated two-input ONNX adapter for the
256px model, because InsightFace 0.7.3 otherwise misclassifies the latter as
an ArcFace recognition model.
Set `VISENLO_MODELS=/another/path` to use another model directory
(`REACTORX_MODELS` still works as a legacy fallback).

### Getting the models — Model Setup Guide

Model weights are **not included in this repository** (size and license
restrictions) and **are not auto-downloaded** (licensing). The `buffalo_l` analysis pack and swap weights require explicit setup. Run the helper with `--download` after reviewing the upstream licenses; the pipeline never downloads missing analysis models during a swap. The app shows a live checker:

```
[✓/✗] buffalo_l      — Face analysis (5 files, required)
[✓/✗] inswapper_128  — Main swap (required, ~529 MB)
[✓/✗] BiSeNet        — Face parsing (optional)
[✓/✗] XSeg           — Occlusion (~68 MB)
[✓/✗] CodeFormer     — Restoration (~377 MB)
          ↓ required assets ✓ → [Run identity swap]
```

**Install missing models** (from your `Visenlo/` folder):

```bash
BASE=https://huggingface.co/facefusion/models-3.0.0/resolve/main
mkdir -p models
curl -L -o models/bisenet_resnet_34.onnx $BASE/bisenet_resnet_34.onnx   # face parsing, MIT
curl -L -o models/codeformer.onnx     $BASE/codeformer.onnx            # restoration, CC BY-NC 4.0 (alt: models/restoration/codeformer.onnx)
curl -L -o models/inswapper_128.onnx  $BASE/inswapper_128.onnx         # swap model, research-only
# optional occlusion mask (GPL-3.0):
curl -L -o models/xseg_1.onnx https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx
# buffalo_l manual alt: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -> models/insightface/models/buffalo_l/
python scripts/download_models.py --check   # verify — should show all ✓
# If you accept licenses and want auto-fetch: python scripts/download_models.py --download
```

`reswapper_256.onnx` is optional and has no canonical public host; without it
the 256px dropdown entry is unavailable. Note that `inswapper_128` and
`reswapper_256` are InsightFace research models — non-commercial use only.
Pixel-boost suffixes (`@256` etc.) are only valid for `inswapper_128`; the UI
rejects them for `reswapper_256` (see Pixel Boost section above).

## Face restoration (CodeFormer, GFPGAN, GPEN, RestoreFormer++)

Guarded restoration stage (CodeFormer by default when installed). Pick the
**Face restorer** in Fine-tune; `scripts/download_models.py --download --all`
fetches all four ONNX graphs from the FaceFusion model hub.

| Restorer | File | Character |
|---|---|---|
| CodeFormer | `codeformer.onnx` (~377 MB, CC BY-NC 4.0) | strongest cleanup, can "beautify" away identity — the guard catches this |
| GFPGAN 1.4 | `gfpgan_1.4.onnx` (~340 MB, Apache-2.0) | balanced, keeps identity better |
| GPEN BFR 512 | `gpen_bfr_512.onnx` (~280 MB, Apache-2.0) | most faithful, mild texture recovery |
| RestoreFormer++ | `restoreformer_plus_plus.onnx` (Apache-2.0) | transformer prior, good on heavy blur |

- **Restoration blend** (0..1, default 0.7) mixes the restored crop into the
  swapped crop before paste-back; 1.0 is full replacement.
- **CodeFormer fidelity** (`w`) only applies to CodeFormer.
- The selected candidate is re-aligned to the 512px FFHQ template, restored,
  blended, pasted through the landmark mask on the face ROI, then finished. The
  identity guard embeds the face at the target landmarks before/after and
  rejects a drop larger than `likeness_tolerance`; the report shows why
  (`restoration: off (rejected 0.558->0.536)`).

Restorers are loaded once each and kept in memory.

The engine requests InsightFace's 106-point landmark output when available and
falls back conservatively when a runtime only exposes five points. Skin-only
eligibility masks preserve hair, clothing, accessories, body, and background.

This app is intended for images you own or have permission to edit.

## Quality stack

- **Best likeness (default).** Runs unchanged `inswapper_128` at native, 256px,
  and 512px pixel-boost settings (see Pixel Boost above; boosts beyond the
  face's native resolution are skipped), scores each composited face in crop
  space, and keeps the sharpest candidate within 0.01 of the best score. Since
  v2 it costs roughly one swap plus the extra tiles — no re-detection.
  “Reference similarity” is an ArcFace comparison value, not a guarantee that
  an image looks convincing. Selecting an explicit `@256`, `@512`, `@1024`, or
  `@2048` entry always runs that exact boost in either quality mode.
- **Mouth handling.** The default swaps the reference lips but preserves only
  the target teeth/open-mouth interior. Choose **Keep entire target mouth** for
  expression stability or **Swap entire mouth** when likeness matters more than
  possible teeth artifacts. Teeth protection requires the optional parser.
- **Scene protection.** XSeg occlusion recovery and parsed face-region masks
  keep target hair, body, scene, and other faces out of color correction and
  enhancement while preserving the swapper's own feathered identity blend.
- **GPU acceleration.** The launcher selects a CPU or GPU ONNX Runtime build.
  GPU operation also requires compatible NVIDIA drivers and CUDA/cuDNN libraries. Without a GPU it falls
  back to CPU transparently. If the CUDA provider is advertised but
  `libcublas`/`libcublasLt` is missing (seen as
  `Failed to load ... libonnxruntime_providers_cuda.so`), the pipeline now
  automatically falls back to `CPUExecutionProvider` and retries model load
  instead of failing permanently.
- **Pixel boost.** `inswapper_128@256` / `inswapper_128@512` / `@1024` / `@2048`
  run the 128px model on polyphase tiles of a larger aligned crop (FaceFusion
  technique), yielding 2x/4x/8x/16x sharper swaps with the same identity
  embedding. Input size is detected robustly for both `NCHW` and `NHWC` exports.
  Full details — including tile counts, costs, and Best-likeness comparison
  logic — are in **Pixel Boost — explicit guide** above.
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

Parsing, occlusion masking, and restoration are optional; the pipeline logs
and falls back when they are unavailable. Analysis and the selected swap model are required.

## What stays private

Nothing that touches your photos is tracked by git: `models/` (weights and the
identity bank `models/identities/*.npz`), `outputs/`, `testdata/`,
`test_images/`, `.gradio/` uploads, logs and `.env*` are all ignored. Settings
presets live outside the repo in `~/.config/visenlo/`. The source ZIP built by
`scripts/make_dist.py` applies the same exclusions.

## Self-check, tests & model helper

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q                   # unit tests, no weights needed
.venv/bin/python scripts/selfcheck.py                # 40+ checks, synthetic data only
.venv/bin/python scripts/bench_swap.py --match index  # timed real swap on testdata/
.venv/bin/python scripts/download_models.py --check   # checker → guide (no download, default)
.venv/bin/python scripts/download_models.py --download  # actually fetch (only if you accept licenses)
python launcher.py --skip-check                     # skip checks for faster launch
python launcher.py --host 0.0.0.0 --auth me:secret  # LAN access behind a login
```

`requirements.txt` is the CPU runtime; the launcher installs
`requirements-gpu.txt` (adds `onnxruntime-gpu`) when an NVIDIA GPU is found.

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

## Alternative swap models

The face-model menu now includes AlphaFace, BlendSwap, GHOST 1/2/3, HifiFace,
HyperSwap 1a/1b/1c, SimSwap 256/512, UniFace, InSwapper FP16, MobileFaceSwap,
and FaceDancer variants alongside the original InSwapper and ReSwapper.
The selected model's weights and optional runtime determine launch readiness.
InSwapper remains the default; masking, mouth controls and restoration remain
Visenlo's existing finishing stages.

```bash
.venv/bin/python scripts/swap_models.py --all                 # check
.venv/bin/python scripts/swap_models.py --download --all      # download verified public assets
.venv/bin/python scripts/setup_mobilefaceswap.py             # optional isolated Paddle runtime + weights
.venv/bin/python scripts/setup_facedancer.py                 # optional isolated TensorFlow runtime
```

FaceDancer's official generator weights require Hugging Face access approval and
are not bundled. Its backend has not been validated with real generator weights.
MobileFaceSwap is native 224px only; unsupported boost sizes are not silently
replaced with resizing. New model `@` suffixes use that model's native resolution.
Image-conditioned models use the highest-quality accepted reference.

See [model inventory and remaining integration work](docs/SWAP_MODELS.md) for
sources, model restrictions and research backends that are not yet implemented.
Run `scripts/compare_swappers.py --all` with the project Python to save actual
local comparisons, then `scripts/report_swappers.py` to build their HTML report.
