"""Pipeline selection logic without loading any model."""
import numpy as np
import pytest

from visenlo.engine import FaceRecord
from visenlo.pipeline import (Candidate, FaceAssignment, PipelineConfig, ReferenceSet, VisenloPipeline,
                              candidate_boosts_for, parse_swapper_spec, select_likeness_candidate,
                              select_target_record)


def _face(gender=None, embedding=None, x=0):
    rec = FaceRecord(face=None, bbox=(x, 0, x + 10, 10), landmarks=np.zeros((5, 2), np.float32),
                     embedding=embedding, score=.9)
    rec.gender = gender
    return rec


def test_parse_swapper_spec():
    assert parse_swapper_spec("inswapper_128.onnx") == ("inswapper_128.onnx", 1)
    assert parse_swapper_spec("inswapper_128.onnx@512") == ("inswapper_128.onnx", 4)
    assert parse_swapper_spec("inswapper_128@256") == ("inswapper_128.onnx", 2)
    with pytest.raises(ValueError):
        parse_swapper_spec("inswapper_128.onnx@300")


def test_candidate_boosts():
    assert candidate_boosts_for("inswapper_128.onnx", "inswapper_128.onnx", 1, "best_likeness") == (1, 2, 4)
    assert candidate_boosts_for("inswapper_128.onnx@512", "inswapper_128.onnx", 4, "best_likeness") == (4,)
    assert candidate_boosts_for("inswapper_128.onnx", "inswapper_128.onnx", 1, "manual") == (1,)


def test_select_prefers_sharpest_within_tolerance():
    probe = np.zeros((1, 1, 3), np.uint8)
    cands = [Candidate(1, "default", probe, .80, .8, 100.0),
             Candidate(2, "default", probe, .805, .8, 140.0),
             Candidate(4, "default", probe, .798, .8, 90.0)]
    assert select_likeness_candidate(cands).boost == 2
    # Clear likeness gain beats sharpness.
    cands.append(Candidate(1, "identity-preserving", probe, .83, .8, 10.0))
    assert select_likeness_candidate(cands).profile == "identity-preserving"
    # Legacy 5-tuples still work and tie-break to the higher boost.
    chosen = select_likeness_candidate([(1, "default", probe, .81, .8), (2, "default", probe, .81, .8)])
    assert chosen[0] == 2
    assert select_likeness_candidate([(1, "default", probe, None, None)])[0] == 1


def test_select_target_modes():
    e1 = np.array([1, 0, 0], np.float32)
    e2 = np.array([0, 1, 0], np.float32)
    f = _face(0, e1, 0)
    m = _face(1, e2, 20)
    assert select_target_record([f, m], "index", 1) is m
    assert select_target_record([m, f], "gender", 0, 0) is f
    assert select_target_record([f, m], "similar", anchor=e2) is m
    assert select_target_record([f, m], "similar", anchor=np.array([0, 0, 1], np.float32)) is None
    assert select_target_record([f, m], "similar", anchor=e2, exclude=[m]) is None
    with pytest.raises(ValueError):
        select_target_record([f], "index", 3)


def test_resolve_assignments_multi_person(tmp_path):
    pipe = VisenloPipeline(str(tmp_path / "models"), PipelineConfig(save_swaps=False))
    e1 = np.array([1, 0, 0], np.float32)
    e2 = np.array([0, 1, 0], np.float32)
    a = _face(0, e1, 0)
    b = _face(1, e2, 50)
    ref_a = ReferenceSet(e1, [_face(0, e1)], [], "A")
    ref_b = ReferenceSet(e2, [_face(1, e2)], [], "B")
    plan = pipe.resolve_assignments([a, b], [FaceAssignment(ref_a, "index", 0), FaceAssignment(ref_b, "index", 1)])
    assert [(r is a, s.label) for r, s in plan] == [(True, "A"), (False, "B")]
    plan = pipe.resolve_assignments([a, b], [FaceAssignment(ref_b, "gender"), FaceAssignment(ref_a, "all")])
    assert plan[0][0] is b and plan[1][0] is a and len(plan) == 2
    with pytest.raises(ValueError, match="assigned twice"):
        pipe.resolve_assignments([a, b], [FaceAssignment(ref_a, "index", 0), FaceAssignment(ref_b, "index", 0)])
    with pytest.raises(ValueError, match="gender"):
        pipe.resolve_assignments([a], [FaceAssignment(ref_b, "gender")])


def test_config_defaults():
    cfg = PipelineConfig()
    assert cfg.quality_mode == "best_likeness"
    assert cfg.mask_mode == "region" and cfg.blend_mode == "multiband"
    assert cfg.restorer == "codeformer" and 0 < cfg.restoration_blend <= 1
    assert not cfg.boost_dealias and cfg.refine_landmarks


def test_invalid_inputs_rejected_before_model_load(tmp_path):
    pipe = VisenloPipeline(str(tmp_path / "models"))
    with pytest.raises(ValueError):
        pipe.process([np.zeros((8, 8, 3), np.uint8)], np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError):
        pipe.swap(np.zeros((8, 8, 3), np.uint8), [])
    assert pipe._analysis is None
