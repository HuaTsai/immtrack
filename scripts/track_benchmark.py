"""Run BBoxTracker over the benchmark detection sequence.

Reads detections from ``benchmark/detections/<ts_us>.txt`` (one frame per
file, sorted by timestamp), feeds them through ``immtrack.BBoxTracker``,
and writes confirmed tracks to ``benchmark/tracking/<ts_us>.txt``.

Detection file format (per line, whitespace-separated):
    x y z l w h rot class score

Output file format (same fields plus track_id at line end):
    x y z l w h rot class score track_id

The output uses the tracker's filtered state (UKF-smoothed pose, EMA
size). Frames with no confirmed tracks still produce an empty file so
every input timestamp has a 1:1 output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import immtrack


REPO_ROOT = Path(__file__).resolve().parent.parent
DETECTIONS_DIR = REPO_ROOT / "benchmark" / "detections"
TRACKING_DIR = REPO_ROOT / "benchmark" / "tracking"


def parse_detection_file(
    path: Path,
    min_score: float,
    per_class_min_score: dict[str, float],
) -> list[immtrack.BoundingBox]:
    boxes: list[immtrack.BoundingBox] = []
    with path.open() as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 9:
                score = float(parts[8])
            elif len(parts) == 8:
                # No score column; treat as perfect detection.
                score = 1.0
            else:
                raise ValueError(
                    f"{path}:{lineno}: expected 8 or 9 fields, got {len(parts)}: {line!r}"
                )
            x, y, z, l, w, h, rot = (float(v) for v in parts[:7])
            class_name = parts[7]
            cutoff = per_class_min_score.get(class_name, min_score)
            if score < cutoff:
                continue
            boxes.append(
                immtrack.BoundingBox(
                    x=x, y=y, z=z, l=l, w=w, h=h, rot=rot,
                    class_name=class_name, score=score,
                )
            )
    return boxes


def write_tracking_file(path: Path, tracks: list[immtrack.TrackedObject]) -> None:
    with path.open("w") as f:
        for t in tracks:
            f.write(
                f"{t.x:.6f} {t.y:.6f} {t.z:.6f} "
                f"{t.l:.6f} {t.w:.6f} {t.h:.6f} {t.rot:.6f} "
                f"{t.class_name} {t.score:.6f} {t.id}\n"
            )


def run(
    detections_dir: Path,
    tracking_dir: Path,
    n_init: int,
    max_age: int,
    min_hits: int,
    size_ema_alpha: float,
    min_score: float,
    per_class_min_score: dict[str, float],
) -> None:
    if not detections_dir.is_dir():
        raise FileNotFoundError(f"detections dir not found: {detections_dir}")
    tracking_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(detections_dir.glob("*.txt"))
    if not files:
        raise RuntimeError(f"no detection files in {detections_dir}")

    tracker = immtrack.BBoxTracker(
        n_init=n_init,
        max_age=max_age,
        min_hits=min_hits,
        size_ema_alpha=size_ema_alpha,
    )

    prev_ts_us: int | None = None
    total_track_lines = 0

    if per_class_min_score:
        print(f"score filter: default={min_score} per-class={per_class_min_score}")
    elif min_score > 0.0:
        print(f"score filter: default={min_score}")

    for idx, det_path in enumerate(files):
        ts_us = int(det_path.stem)
        detections = parse_detection_file(det_path, min_score, per_class_min_score)

        if prev_ts_us is None:
            tracks = tracker.update(detections)
        else:
            dt = (ts_us - prev_ts_us) / 1_000_000.0
            if dt <= 0.0:
                raise RuntimeError(
                    f"non-monotonic timestamps: {prev_ts_us} -> {ts_us} in {det_path}"
                )
            tracks = tracker.update(detections, dt)
        prev_ts_us = ts_us

        out_path = tracking_dir / det_path.name
        write_tracking_file(out_path, tracks)
        total_track_lines += len(tracks)

        if (idx + 1) % 25 == 0 or idx == len(files) - 1:
            print(
                f"[{idx + 1:>3}/{len(files)}] {det_path.name} "
                f"detections={len(detections):>3} tracks={len(tracks):>3} "
                f"alive={tracker.track_count():>3}"
            )

    print(
        f"done: {len(files)} frames, {total_track_lines} confirmed-track lines, "
        f"output -> {tracking_dir}"
    )


def parse_per_class_min_score(values: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in values:
        for piece in item.split(","):
            piece = piece.strip()
            if not piece:
                continue
            if "=" not in piece:
                raise argparse.ArgumentTypeError(
                    f"--min-score-class expects CLASS=FLOAT, got {piece!r}"
                )
            cls, val = piece.split("=", 1)
            out[cls.strip()] = float(val)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detections", type=Path, default=DETECTIONS_DIR)
    p.add_argument("--tracking", type=Path, default=TRACKING_DIR)
    p.add_argument("--n-init", type=int, default=3)
    p.add_argument("--max-age", type=int, default=5)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--size-ema-alpha", type=float, default=0.7)
    p.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="drop detections with score < value before tracker.update (default 0)",
    )
    p.add_argument(
        "--min-score-class",
        action="append",
        default=[],
        help="per-class override, e.g. 'pedestrian=0.7' (repeatable or comma-separated)",
    )
    args = p.parse_args(argv)

    per_class = parse_per_class_min_score(args.min_score_class)

    run(
        args.detections,
        args.tracking,
        args.n_init,
        args.max_age,
        args.min_hits,
        args.size_ema_alpha,
        args.min_score,
        per_class,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
