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
