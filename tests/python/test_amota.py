"""Smoke test for AMOTA Python binding."""
from __future__ import annotations

from immtrack import BoundingBox
from immtrack.metrics import AmotaConfig, MatchMetric, amota


def gt_box(track_id: int, x: float, y: float, cls: str = "car") -> BoundingBox:
    return BoundingBox(
        x=x,
        y=y,
        z=0.0,
        l=4.0,
        w=2.0,
        h=1.5,
        rot=0.0,
        class_name=cls,
        score=1.0,
        track_id=track_id,
    )


def pred_box(
    track_id: int, x: float, y: float, score: float, cls: str = "car"
) -> BoundingBox:
    return BoundingBox(
        x=x,
        y=y,
        z=0.0,
        l=4.0,
        w=2.0,
        h=1.5,
        rot=0.0,
        class_name=cls,
        score=score,
        track_id=track_id,
    )


def test_perfect_predictions_amota_is_one():
    gt = [
        [gt_box(1, 0, 0)],
        [gt_box(1, 1, 0)],
        [gt_box(1, 2, 0)],
    ]
    pred = [
        [pred_box(100, 0, 0, 0.9)],
        [pred_box(100, 1, 0, 0.9)],
        [pred_box(100, 2, 0, 0.9)],
    ]
    result = amota(gt, pred)
    assert abs(result.overall - 1.0) < 1e-9
    assert abs(result.per_class["car"] - 1.0) < 1e-9


def test_iou3d_metric_via_config():
    cfg = AmotaConfig()
    cfg.metric = MatchMetric.Iou3d
    cfg.match_threshold = 0.5

    gt = [[gt_box(1, 0, 0)]]
    pred = [[pred_box(100, 0.1, 0, 0.9)]]
    result = amota(gt, pred, cfg)
    assert abs(result.per_class["car"] - 1.0) < 1e-9


def test_empty_inputs():
    result = amota([], [])
    assert result.overall == 0.0
    assert dict(result.per_class) == {}
