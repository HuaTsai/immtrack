"""Append GT track ids from ``benchmark/result/<ts>.json`` to the
matching ``benchmark/annotaions/<ts>.txt`` file.

Each annotation row gains a 9th column: the integer ``trackName`` of
the JSON instance whose class + center3D + size3D match the row.
By default the file is rewritten in place.

Class mapping (JSON className -> annotation class):
    Car -> car
    Bus -> bus
    Motorcycle -> motorcycle
    Pedestrian -> pedestrian
    Truck Small -> truck
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ANN_DIR = REPO_ROOT / "benchmark" / "annotaions"
RESULT_DIR = REPO_ROOT / "benchmark" / "result"

CLASS_MAP = {
    "Car": "car",
    "Bus": "bus",
    "Motorcycle": "motorcycle",
    "Pedestrian": "pedestrian",
    "Truck Small": "truck",
}

# Annotation files round to 2 decimals; allow a small tolerance.
POS_TOLERANCE_M = 0.05


def load_json_instances(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected non-empty top-level list")
    out = []
    for inst in data[0].get("instances", []):
        contour = inst["contour"]
        c = contour["center3D"]
        s = contour["size3D"]
        cls_json = inst["className"]
        cls = CLASS_MAP.get(cls_json)
        if cls is None:
            raise ValueError(f"{path}: unknown className {cls_json!r}")
        try:
            track_id = int(inst["trackName"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path}: trackName is not an int: {inst.get('trackName')!r}") from e
        out.append(
            {
                "class": cls,
                "x": float(c["x"]),
                "y": float(c["y"]),
                "z": float(c["z"]),
                "l": float(s["x"]),
                "w": float(s["y"]),
                "h": float(s["z"]),
                "track_id": track_id,
            }
        )
    return out


def parse_annotation_line(line: str) -> tuple[list[str], float, float, float, str]:
    parts = line.split()
    if len(parts) != 8:
        raise ValueError(f"expected 8 fields, got {len(parts)}: {line!r}")
    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
    cls = parts[7]
    return parts, x, y, z, cls


def match_row_to_instance(
    x: float, y: float, z: float, cls: str, instances: list[dict], used: set[int]
) -> int:
    best_idx = -1
    best_dist = math.inf
    for i, inst in enumerate(instances):
        if i in used or inst["class"] != cls:
            continue
        dx = inst["x"] - x
        dy = inst["y"] - y
        dz = inst["z"] - z
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d < best_dist:
            best_dist = d
            best_idx = i
    if best_idx < 0:
        raise RuntimeError(f"no candidate instance for class={cls} at ({x},{y},{z})")
    if best_dist > POS_TOLERANCE_M:
        raise RuntimeError(
            f"closest match for class={cls} at ({x},{y},{z}) is {best_dist:.4f}m away "
            f"(> {POS_TOLERANCE_M}m); JSON misalignment?"
        )
    return best_idx


def integrate_one_file(ann_path: Path, json_path: Path, out_path: Path) -> int:
    instances = load_json_instances(json_path)
    used: set[int] = set()
    out_lines: list[str] = []

    raw_lines = ann_path.read_text().splitlines()
    for lineno, raw in enumerate(raw_lines, 1):
        line = raw.strip()
        if not line:
            out_lines.append("")
            continue
        try:
            parts, x, y, z, cls = parse_annotation_line(line)
            idx = match_row_to_instance(x, y, z, cls, instances, used)
        except (ValueError, RuntimeError) as e:
            raise RuntimeError(f"{ann_path}:{lineno}: {e}") from e
        used.add(idx)
        track_id = instances[idx]["track_id"]
        out_lines.append(" ".join(parts) + f" {track_id}")

    if len(used) != len(instances):
        unused = [
            f"{instances[i]['class']}#{instances[i]['track_id']}"
            for i in range(len(instances))
            if i not in used
        ]
        # Not fatal: JSON may have extra instances filtered out of annotations.
        print(
            f"  WARN {ann_path.name}: {len(instances) - len(used)} JSON instance(s) "
            f"unused ({', '.join(unused)})",
            file=sys.stderr,
        )

    out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return len(used)


def run(ann_dir: Path, result_dir: Path, out_dir: Path) -> None:
    if not ann_dir.is_dir():
        raise FileNotFoundError(ann_dir)
    if not result_dir.is_dir():
        raise FileNotFoundError(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.txt"))
    if not ann_files:
        raise RuntimeError(f"no .txt files in {ann_dir}")

    total_rows = 0
    missing_json: list[str] = []
    for i, ann_path in enumerate(ann_files, 1):
        ts = ann_path.stem
        json_path = result_dir / f"{ts}.json"
        if not json_path.exists():
            missing_json.append(ts)
            continue
        out_path = out_dir / ann_path.name
        n = integrate_one_file(ann_path, json_path, out_path)
        total_rows += n
        if i % 25 == 0 or i == len(ann_files):
            print(f"[{i:>3}/{len(ann_files)}] {ann_path.name}: {n} rows tagged")

    print(f"done: {len(ann_files) - len(missing_json)} files, {total_rows} rows tagged")
    if missing_json:
        print(f"  missing JSON for {len(missing_json)} timestamps: {missing_json[:3]}...")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, default=ANN_DIR)
    p.add_argument("--result", type=Path, default=RESULT_DIR)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output dir (defaults to in-place rewrite of --annotations)",
    )
    args = p.parse_args(argv)
    out_dir = args.output if args.output is not None else args.annotations
    run(args.annotations, args.result, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
