"""End-to-end tracker test using the pybind11 binding."""
from __future__ import annotations

import immtrack


def make_box(
    cls: str,
    x: float,
    y: float,
    z: float = 0.0,
    yaw: float = 0.0,
    score: float = 0.9,
) -> immtrack.BoundingBox:
    return immtrack.BoundingBox(
        x=x,
        y=y,
        z=z,
        l=4.0,
        w=2.0,
        h=1.5,
        rot=yaw,
        class_name=cls,
        score=score,
    )


def test_first_frame_returns_empty():
    tracker = immtrack.BBoxTracker()
    out = tracker.update([make_box("car", 0, 0)])
    assert out == []


def test_confirms_after_three_frames():
    tracker = immtrack.BBoxTracker()
    tracker.update([make_box("car", 0, 0)])
    tracker.update([make_box("car", 1, 0)], dt=0.1)
    out = tracker.update([make_box("car", 2, 0)], dt=0.1)
    assert len(out) == 1
    assert out[0].class_name == "car"
    assert out[0].id >= 0


def test_per_class_strict_no_cross_association():
    tracker = immtrack.BBoxTracker()
    for i in range(3):
        boxes = [
            make_box("car", i * 0.5, 0),
            make_box("pedestrian", i * 0.5, 0),
        ]
        if i == 0:
            tracker.update(boxes)
        else:
            tracker.update(boxes, dt=0.1)
    out = tracker.update(
        [make_box("car", 1.5, 0), make_box("pedestrian", 1.5, 0)], dt=0.1
    )
    classes = sorted(t.class_name for t in out)
    assert classes == ["car", "pedestrian"]


def test_id_persists_across_one_frame_occlusion():
    tracker = immtrack.BBoxTracker()
    tracker.update([make_box("car", 0, 0)])
    tracker.update([make_box("car", 1, 0)], dt=0.1)
    out = tracker.update([make_box("car", 2, 0)], dt=0.1)
    track_id = out[0].id

    tracker.update([], dt=0.1)
    resumed = tracker.update([make_box("car", 4, 0)], dt=0.1)
    assert len(resumed) == 1
    assert resumed[0].id == track_id


def test_reset_clears_tracks():
    tracker = immtrack.BBoxTracker()
    tracker.update([make_box("car", 0, 0)])
    assert tracker.track_count() == 1
    tracker.reset()
    assert tracker.track_count() == 0
