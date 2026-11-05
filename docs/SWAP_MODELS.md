# Swap model research and integration

Research date: 2026-09-21. This is a source-backed inventory, not a claim that
every private checkpoint, derivative, commercial service, or unpublished paper
has been located. Model families and precision/checkpoint variants are distinct.
Internet reviews and authors' quality claims are not used to select a winner.

## Implemented model contracts

The executable inventory is `visenlo/swappers.py`. All models use Visenlo's
existing detection/face selection and finishing stages. Tensor layout, alignment,
source conditioning and inference are backend-specific. No external app pipeline
is imported. Existing InSwapper and ReSwapper inference implementations remain.

| Family | Registered variants | Native crop | Source conditioning |
|---|---|---:|---|
| InSwapper | original, FP16 | 128 | Existing InsightFace adapter |
| ReSwapper | existing local model | 256 | Existing emap adapter |
| AlphaFace | 256 | 256 | Raw buffalo_l embedding |
| BlendSwap | 256 | 256 | Selected reference image |
| GHOST | 1, 2, 3 | 256 | Raw embedding through CrossFace converter |
| HifiFace | unofficial ONNX | 256 | Normalized converted embedding |
| HyperSwap | 1a, 1b, 1c | 256 | Normalized buffalo_l embedding |
| SimSwap | 256, unofficial 512 | 256/512 | Normalized converted embedding |
| UniFace | 256 | 256 | Selected reference image |
| FaceDancer | A, B, B tuned, C, C tuned, D, D tuned, E, c HQ | 256 | Own ArcFace in isolated TensorFlow worker |
| MobileFaceSwap | 224 | 224 | Own ArcFace/features in isolated Paddle worker |

ONNX tensor conventions and release identifiers were checked against
[FaceFusion's registry and inference code](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py).
Weights are distributed under their individual licenses; the application does
not relicense them. HifiFace and UniFace license metadata is unresolved upstream.
The HifiFace and SimSwap 512 conversions are explicitly unofficial exports.

Original sources: [InsightFace](https://github.com/deepinsight/insightface/tree/master/examples/in_swapper),
[ReSwapper](https://github.com/somanchiu/ReSwapper),
[GHOST](https://github.com/ai-forever/ghost),
[SimSwap](https://github.com/neuralchen/SimSwap),
[BlendFace](https://github.com/mapooon/BlendFace),
[FaceDancer](https://github.com/felixrosberg/FaceDancer),
[MobileFaceSwap](https://github.com/Seanseattle/MobileFaceSwap).

## Availability and remaining research backends

| Model/project | Primary evidence | Current disposition |
|---|---|---|
| FaceDancer | [Official weights](https://huggingface.co/felixrosberg/FaceDancer) | Generator download returns HTTP 401 / GatedRepo. Adapter exists; original custom layers import successfully. Real inference requires authorized weights. |
| InfoSwap | [Author's usage instructions](https://github.com/GGGHSL/InfoSwap-master/blob/master/USAGE.md) | Authors require an institutional email request for pretrained models. Not integrated; no request sent. |
| REFace | [Official repository](https://github.com/Sanoojan/REFace), [checkpoint](https://huggingface.co/Sanoojan/REFace/blob/main/last.ckpt) | Public 6.07 GB checkpoint; inference explicitly calls CUDA and uses a separate diffusion/masking stack. Dedicated backend not implemented or validated here. |
| DiffSwap | [Official repository](https://github.com/wl-zhao/DiffSwap) | Author provides checkpoint collection on Tsinghua Cloud; separate 3D/masked-diffusion workflow. Not integrated or validated. |
| DiffFace | [Official repository](https://github.com/hxngiee/DiffFace) | Author links four checkpoints on SharePoint and an older PyTorch runtime. Not integrated or validated. |
| E4S / E4S2024 | [Official repository](https://github.com/e4s2024/E4S2024) | Regional GAN inversion with multiple supporting networks and CUDA setup. Not integrated or validated. |
| FSGAN | [Official repository](https://github.com/YuvalNirkin/fsgan) | Pretrained model downloader and CUDA-oriented multi-network workflow. Not integrated or validated. |
| StyleSwap | [Official repository](https://github.com/Seanseattle/StyleSwap) | Inference code references 256/512 Paddle checkpoints, but README supplies no release/download procedure. No verified generator download found. |
| VFace | [Official repository](https://github.com/Sanoojan/VFace) | Video framework using REFace; not a separate image-swap checkpoint. No video workflow added. |
| Stand-In | [Official repository](https://github.com/WeChatCV/Stand-In) | Identity-preserving video generation, with video swap inference. Outside the existing still-image workflow. |
| DeepFaceLab / Faceswap training architectures | [DeepFaceLab](https://github.com/iperov/DeepFaceLab), [Faceswap](https://github.com/deepfakes/faceswap) | Identity-specific training systems, not universally interchangeable one-reference pretrained swappers. No training workflow added. |

This table deliberately distinguishes unfinished integration from unavailable
weights. GPU unavailability does not prove a model is incompatible. Research
beyond the registered backends remains incomplete, including MegaFS, FaceShifter,
other recent research families, and checkpoint-level auditing of their releases.

## Pixel boost and unchanged features

- Existing InSwapper choices and explicit boost semantics are preserved.
- New fixed-size backends use the existing polyphase decomposition around their
  native crop inference, followed by Visenlo's existing paste-back helper.
- An explicit size must be a multiple of native resolution. It is never silently
  replaced with ordinary image upscaling. MobileFaceSwap exposes native 224 only.
- Best likeness compares native and supported candidates up to 512px. Existing
  InSwapper finishing-profile selection remains InSwapper-specific.
- Mouth/teeth modes, XSeg, parsing, color controls, sharpening, guarded restoration,
  face matching, upload workflow and saved output dimensions remain shared.
- Pixel-conditioned backends use the highest-quality accepted reference (first
  upload breaks ties). Embedding-based backends retain the aggregate identity;
  raw-embedding converters receive that direction at the weighted mean magnitude.

## Installation and verification

```bash
.venv/bin/python scripts/swap_models.py --all
.venv/bin/python scripts/swap_models.py --download --all
.venv/bin/python scripts/setup_facedancer.py
.venv/bin/python scripts/setup_mobilefaceswap.py
MPLCONFIGDIR=/tmp/visenlo-mpl .venv/bin/python scripts/test_swappers.py
MPLCONFIGDIR=/tmp/visenlo-mpl .venv/bin/python scripts/selfcheck.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/visenlo-mpl \
  .venv/bin/python scripts/compare_swappers.py --all
.venv/bin/python scripts/report_swappers.py
```

The downloader resolves each Hugging Face repository to a commit before download,
validates the advertised LFS SHA256, and records its URL/revision/hash alongside
the weight. Existing files are preserved. Partial downloads can be resumed.
FaceDancer requires access through its model owner's normal access procedure;
after approval, the downloader accepts an authorized read token through `HF_TOKEN`.
Tokens are not included in command-line arguments or download receipts.
No unofficial copy is substituted. MobileFaceSwap's author archive has no
published checksum, so its receipt records a local hash without claiming
independent verification. Optional runtimes stay under `.venvs/`.

Comparison outputs, logs, exact settings, input hashes, and per-run statuses live
in `outputs/model-comparison/`. The HTML report shows actual successful outputs.
Runs use CPU explicitly because this session cannot access NVIDIA CUDA. Missing
weights and timeouts are not passing tests. One or two fixtures cannot establish
a generally best model, and ArcFace similarity alone does not measure visual
quality. Keep InSwapper as the default pending broader visual evaluation.

## Verified in this checkout

- All 16 installed models completed native-resolution CPU inference on both
  fixtures, including all 14 new runnable alternatives. FaceDancer's nine entries
  are experimental adapters with missing gated generator weights, not nine
  additional verified models.
- The unchanged self-check suite passes. New contract tests cover registry
  resolution, explicit boost precedence, required converters, alignment, tensor
  dtypes/ranges, identity conditioning and selected-model readiness.
- Source-level comparisons confirm the existing detector, target selection,
  finishing, guarded restoration, ReSwapper adapter, configuration defaults,
  engine module and original boost module are unchanged.
- Boosted inference passes for AlphaFace, BlendSwap, GHOST 1, HifiFace, HyperSwap 1a,
  SimSwap 256, SimSwap 512, and UniFace. SimSwap 512 was checked at 1024px; the
  others at 512px. HyperSwap's automatic Best likeness selection also passes.
  The original 2048px InSwapper test
  exceeded its 300-second CPU budget; its full real-image result is unverified.
- Model switching HyperSwap → InSwapper → HyperSwap passes in one pipeline with
  two references and unchanged configuration.
- The untouched right half of the two-face fixture was pixel-identical for
  every output examined; no second-face changes were observed.
- Timings include concurrent CPU checks and are not controlled speed benchmarks.

Early native identity scores illustrate why no automatic replacement was made:

| Model | First pair | Second pair |
|---|---:|---:|
| InSwapper 128 | 0.592 | 0.814 |
| AlphaFace 256 | 0.635 | 0.767 |
| GHOST 2 | 0.437 | 0.757 |
| HyperSwap 1a | 0.566 | 0.688 |

These are Visenlo's ArcFace similarities after shared finishing. AlphaFace is
worth comparing visually with InSwapper, but its second-fixture output still
shows a colored face region against the grayscale target. UniFace's first
output shows pronounced dark facial shading. A higher identity score does not
establish better blending, expression, teeth, or overall realism.
