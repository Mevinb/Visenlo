"""Headless command line: ``python -m visenlo <command> ...``.

Commands
  swap      swap one identity into one or more images
  video     swap one identity into a video file
  faces     list detected faces (index, bbox, gender, age) of an image
  identity  save / list / delete identities in the identity bank
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _models_dir(args):
    return Path(args.models or os.environ.get("VISENLO_MODELS", os.environ.get("REACTORX_MODELS",
                                                                                 Path(__file__).resolve().parents[1] / "models")))


def _read(path):
    data = np.fromfile(str(path), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot read image: {path}")
    return image


def _config(args):
    from .pipeline import PipelineConfig
    cfg = PipelineConfig(save_swaps=False)
    cfg.quality_mode = "manual" if args.manual else "best_likeness"
    cfg.codeformer_enabled = not args.no_restore
    if args.restorer:
        cfg.restorer = args.restorer
    if args.restore_blend is not None:
        cfg.restoration_blend = args.restore_blend
    if args.mask:
        cfg.mask_mode = args.mask
    if args.blend:
        cfg.blend_mode = args.blend
    if args.mouth:
        cfg.mouth_mode = args.mouth
    if args.keep_eyes:
        cfg.eye_mode = "keep_target_eyes"
    if args.precision:
        cfg.inswapper_precision = args.precision
    return cfg


def _add_common(parser):
    parser.add_argument("--models", help="models directory (default: ./models or VISENLO_MODELS)")
    parser.add_argument("--ref", action="append", default=[], help="reference image (repeat for several photos)")
    parser.add_argument("--identity", help="use a saved identity from the bank instead of --ref")
    parser.add_argument("--model", default="inswapper_128.onnx", help="swap model, e.g. inswapper_128.onnx@512")
    parser.add_argument("--match", default="index", choices=("index", "gender", "all"))
    parser.add_argument("--face", type=int, default=0, help="target face index (left to right)")
    parser.add_argument("--ref-face", type=int, default=0, help="face index inside the reference image(s)")
    parser.add_argument("--manual", action="store_true", help="run the selected model once (skip Best likeness)")
    parser.add_argument("--no-restore", action="store_true", help="disable face restoration")
    parser.add_argument("--restorer", choices=("codeformer", "gfpgan_1.4", "gpen_bfr_512", "restoreformer_plus_plus"))
    parser.add_argument("--restore-blend", type=float)
    parser.add_argument("--mask", choices=("box", "region"))
    parser.add_argument("--blend", choices=("alpha", "multiband"))
    parser.add_argument("--mouth", choices=("swap_lips_keep_teeth", "keep_target_mouth", "swap_mouth"))
    parser.add_argument("--keep-eyes", action="store_true")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16"))
    parser.add_argument("--cpu", action="store_true", help="force CPU execution")


def _reference_set(pipeline, args):
    from .identity import default_bank
    from .pipeline import FaceAssignment
    if args.identity:
        refset = default_bank(str(pipeline.models_path)).as_reference_set(args.identity)
    else:
        if not args.ref:
            raise SystemExit("Provide --ref IMAGE (repeatable) or --identity NAME")
        refset = pipeline.analyze_references([_read(p) for p in args.ref], args.ref_face, label=Path(args.ref[0]).stem)
    return [FaceAssignment(refset, args.match, args.face)]


def cmd_swap(args):
    from .pipeline import VisenloPipeline
    if args.cpu:
        os.environ["VISENLO_DEVICE"] = "cpu"
    pipeline = VisenloPipeline(str(_models_dir(args)), _config(args))
    assignments = _reference_set(pipeline, args)
    targets = list(args.target)
    out_dir = Path(args.out) if args.out and (len(targets) > 1 or Path(args.out).is_dir()) else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for path in targets:
        try:
            output = pipeline.swap(_read(path), assignments, args.model)
        except Exception as exc:
            failures += 1
            print(f"{path}: FAILED - {exc}", file=sys.stderr)
            continue
        if out_dir:
            dest = out_dir / f"{Path(path).stem}_swapped.png"
        elif args.out:
            dest = Path(args.out)
        else:
            dest = Path(path).with_name(f"{Path(path).stem}_swapped.png")
        ok, encoded = cv2.imencode(dest.suffix or ".png", output.image)
        if not ok:
            raise SystemExit(f"Could not encode {dest}")
        encoded.tofile(str(dest))
        print(f"{path} -> {dest}\n  {output.status}")
    return 1 if failures else 0


def cmd_video(args):
    from .pipeline import VisenloPipeline
    from .video import swap_video
    if args.cpu:
        os.environ["VISENLO_DEVICE"] = "cpu"
    pipeline = VisenloPipeline(str(_models_dir(args)), _config(args))
    assignments = _reference_set(pipeline, args)
    out = args.out or str(Path(args.input).with_name(f"{Path(args.input).stem}_swapped.mp4"))

    def report(p):
        print(f"\r  frame {p.frame}/{p.total}  faces {p.swapped_faces}  {p.elapsed:.0f}s", end="", flush=True)

    result = swap_video(pipeline, args.input, out, assignments, args.model, progress=report,
                        smoothing=args.smoothing, every=args.every, end_frame=args.max_frames)
    print(f"\n{result.output_path}: {result.frames_done} frames, {result.swapped_faces} faces, "
          f"{result.seconds:.0f}s, audio={'yes' if result.audio_copied else 'no'}")
    for note in result.notes:
        print("  note:", note)
    return 0


def cmd_faces(args):
    from .pipeline import VisenloPipeline, gender_label
    pipeline = VisenloPipeline(str(_models_dir(args)))
    for record_index, record in enumerate(pipeline.detect_faces(_read(args.image))):
        x1, y1, x2, y2 = record.bbox
        print(f"face {record_index}: bbox=({x1},{y1},{x2},{y2}) size={x2 - x1}x{y2 - y1} "
              f"gender={gender_label(record.gender)} age={record.age} quality={record.quality:.2f}")
    return 0


def cmd_identity(args):
    from .identity import default_bank
    bank = default_bank(str(_models_dir(args)))
    if args.action == "list":
        for name in bank.names():
            saved = bank.load(name)
            print(f"{name}: {saved.gender_label} age~{saved.age} refs={saved.references} q={saved.quality:.2f}")
        return 0
    if args.action == "delete":
        print("deleted" if bank.delete(args.name) else "not found")
        return 0
    from .pipeline import VisenloPipeline
    if not args.ref:
        raise SystemExit("identity save needs --ref IMAGE (repeatable)")
    pipeline = VisenloPipeline(str(_models_dir(args)))
    refset = pipeline.analyze_references([_read(p) for p in args.ref], args.ref_face, label=args.name)
    saved = bank.save(args.name, refset)
    print(f"saved {saved.name} -> {saved.path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="visenlo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    swap = sub.add_parser("swap", help="swap faces in images")
    _add_common(swap)
    swap.add_argument("--target", action="append", required=True, help="target image (repeatable)")
    swap.add_argument("--out", help="output file (single target) or directory")
    swap.set_defaults(func=cmd_swap)

    video = sub.add_parser("video", help="swap faces in a video")
    _add_common(video)
    video.add_argument("--input", required=True)
    video.add_argument("--out")
    video.add_argument("--smoothing", type=float, default=.5)
    video.add_argument("--every", type=int, default=1, help="swap every Nth frame (others are copied)")
    video.add_argument("--max-frames", type=int)
    video.set_defaults(func=cmd_video)

    faces = sub.add_parser("faces", help="list faces in an image")
    faces.add_argument("image")
    faces.add_argument("--models")
    faces.set_defaults(func=cmd_faces)

    identity = sub.add_parser("identity", help="manage the identity bank")
    identity.add_argument("action", choices=("save", "list", "delete"))
    identity.add_argument("name", nargs="?")
    identity.add_argument("--ref", action="append", default=[])
    identity.add_argument("--ref-face", type=int, default=0)
    identity.add_argument("--models")
    identity.set_defaults(func=cmd_identity)

    args = parser.parse_args(argv)
    if args.command == "identity" and args.action in ("save", "delete") and not args.name:
        parser.error("identity save/delete needs a NAME")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
