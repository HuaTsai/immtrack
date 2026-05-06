"""Evaluate AMOTA on the benchmark sequence.

Compares ``benchmark/tracking/<ts>.txt`` (predictions, 10 fields) against
``benchmark/annotaions/<ts>.txt`` (GT, 9 fields with track_id appended
by ``integrate_track_ids.py``).

GT line format:    x y z l w h rot class track_id
Pred line format:  x y z l w h rot class score track_id
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import immtrack
from immtrack.metrics import AmotaConfig, MatchMetric, amota

REPO_ROOT = Path(__file__).resolve().parent.parent
ANN_DIR = REPO_ROOT / "benchmark" / "annotaions"
PRED_DIR = REPO_ROOT / "benchmark" / "tracking"


def parse_gt_file(path: Path) -> list[immtrack.BoundingBox]:
    boxes: list[immtrack.BoundingBox] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 9:
            raise ValueError(f"{path}:{lineno}: expected 9 fields (GT), got {len(parts)}")
        x, y, z, l, w, h, rot = (float(v) for v in parts[:7])
        boxes.append(
            immtrack.BoundingBox(
                x=x, y=y, z=z, l=l, w=w, h=h, rot=rot,
                class_name=parts[7], score=1.0, track_id=int(parts[8]),
            )
        )
    return boxes


def parse_pred_file(path: Path) -> list[immtrack.BoundingBox]:
    boxes: list[immtrack.BoundingBox] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 10:
            raise ValueError(f"{path}:{lineno}: expected 10 fields (pred), got {len(parts)}")
        x, y, z, l, w, h, rot = (float(v) for v in parts[:7])
        boxes.append(
            immtrack.BoundingBox(
                x=x, y=y, z=z, l=l, w=w, h=h, rot=rot,
                class_name=parts[7], score=float(parts[8]), track_id=int(parts[9]),
            )
        )
    return boxes


def collect_frames(ann_dir: Path, pred_dir: Path):
    ann_files = {p.stem: p for p in ann_dir.glob("*.txt")}
    pred_files = {p.stem: p for p in pred_dir.glob("*.txt")}
    common = sorted(set(ann_files) & set(pred_files))
    if not common:
        raise RuntimeError("no overlapping timestamps between GT and pred")

    only_gt = sorted(set(ann_files) - set(pred_files))
    only_pred = sorted(set(pred_files) - set(ann_files))
    if only_gt:
        print(f"  WARN {len(only_gt)} GT-only frames will be skipped (e.g. {only_gt[:2]})",
              file=sys.stderr)
    if only_pred:
        print(f"  WARN {len(only_pred)} pred-only frames will be skipped (e.g. {only_pred[:2]})",
              file=sys.stderr)

    gt_frames = [parse_gt_file(ann_files[ts]) for ts in common]
    pred_frames = [parse_pred_file(pred_files[ts]) for ts in common]
    return common, gt_frames, pred_frames


def evaluate(ann_dir: Path, pred_dir: Path, metric: MatchMetric, threshold: float) -> None:
    timestamps, gt_frames, pred_frames = collect_frames(ann_dir, pred_dir)

    n_gt = sum(len(f) for f in gt_frames)
    n_pred = sum(len(f) for f in pred_frames)
    gt_ids = {b.track_id for f in gt_frames for b in f}
    pred_ids = {b.track_id for f in pred_frames for b in f}
    print(f"frames    : {len(timestamps)}")
    print(f"GT boxes  : {n_gt}  (unique ids: {len(gt_ids)})")
    print(f"pred boxes: {n_pred}  (unique ids: {len(pred_ids)})")
    print()

    cfg = AmotaConfig()
    cfg.metric = metric
    cfg.match_threshold = threshold
    print(f"config    : metric={metric.name} threshold={threshold} "
          f"recall={list(cfg.recall_values)}")
    print()

    result = amota(gt_frames, pred_frames, cfg)
    print(f"AMOTA overall: {result.overall:.4f}")
    print("AMOTA per-class:")
    for cls in sorted(result.per_class.keys()):
        cls_gt = sum(1 for f in gt_frames for b in f if b.class_name == cls)
        print(f"  {cls:<12}  {result.per_class[cls]:.4f}   (GT={cls_gt})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, default=ANN_DIR)
    p.add_argument("--predictions", type=Path, default=PRED_DIR)
    p.add_argument("--metric", choices=["center", "iou3d"], default="center")
    p.add_argument("--threshold", type=float, default=2.0,
                   help="match threshold: meters for center, 1-IoU for iou3d")
    args = p.parse_args(argv)

    metric = MatchMetric.CenterDistance if args.metric == "center" else MatchMetric.Iou3d
    evaluate(args.annotations, args.predictions, metric, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
