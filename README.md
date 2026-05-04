# immtrack

IMM-UKF multiple object tracking for 3D bounding boxes.

## Setup

After cloning, bootstrap the environment in two steps:

```bash
uv sync --no-install-project
uv sync
```

The first call provisions `scikit-build-core` and `pybind11` so the second call has a build backend available — required because `pyproject.toml` sets `no-build-isolation-package = ["immtrack"]`.

If the editable build directory (`build/`) was deleted later, force a rebuild:

```bash
uv sync --reinstall-package immtrack
```

## Usage

```python
import immtrack

tracker = immtrack.BBoxTracker(n_init=3, max_age=5)

# First frame
boxes_t0 = [
    immtrack.BoundingBox(
        x=0, y=0, z=0, l=4, w=2, h=1.5, rot=0,
        class_name="car", score=0.9,
    ),
]
tracker.update(boxes_t0)

# Subsequent frames pass dt
boxes_t1 = [
    immtrack.BoundingBox(
        x=1, y=0, z=0, l=4, w=2, h=1.5, rot=0,
        class_name="car", score=0.9,
    ),
]
confirmed = tracker.update(boxes_t1, dt=0.1)
for t in confirmed:
    print(t.id, t.x, t.y, t.vx, t.vy)
```

## Evaluate AMOTA

```python
from immtrack import BoundingBox
from immtrack.metrics import AmotaConfig, amota

# gt[f] and pred[f] are lists of BoundingBox for frame f.
# track_id field must be filled on both sides.
result = amota(gt, pred, AmotaConfig())
print(result.overall)
print(dict(result.per_class))
```

## Run Python tests

```bash
uv run pytest tests/python -v
```

## Run C++ unit tests

Ninja is required (`apt install ninja-build` on Debian/Ubuntu).

```bash
cmake -S . -B build-test -G Ninja -DIMMTRACK_BUILD_TESTS=ON
cmake --build build-test
ctest --test-dir build-test --output-on-failure
```

`IMMTRACK_BUILD_TESTS` defaults to `OFF`, so `pip install` / `uv sync` does not download Catch2.
