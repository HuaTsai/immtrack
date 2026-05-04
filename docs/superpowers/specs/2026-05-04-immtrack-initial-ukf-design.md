# Initial UKF Design — `PosVxyzYawCV` + `PosYawObs`

- Status: Approved (pending implementation)
- Date: 2026-05-04
- Scope: First real UKF implementation in `immtrack`. Replaces the v0.1 placeholder scaffold.

## Goals

1. Provide a fully working Unscented Kalman Filter for 3D LiDAR object tracking with:
   - State `[x, y, z, vx, vy, vz, θ]` (7D)
   - Measurement `[x, y, z, θ]` (4D)
   - Constant-velocity dynamics
2. Establish a trait-based, statically-dispatched template framework that scales to additional motion models, observation models, and (later) IMM combinations.
3. Optimize for compile-time specialization — no virtual dispatch, all matrices fixed-size on the stack.

## Non-Goals (this spec)

- IMM (Interacting Multiple Models) integration — separate spec.
- Additional filter back-ends (EKF, Particle Filter) — separate spec.
- Track lifecycle management, data association, gating — belongs to a tracker layer above the filter.
- Multi-model state mixing.
- Production-tuned process/measurement noise — placeholders only.
- Multi-threading or SIMD beyond what Eigen does automatically.
- Filter state serialization / save-load.
- Python `__repr__` / `__str__` formatting beyond defaults.

## Architecture Overview

```text
+-----------------------------------------------------+
|              UKF<Motion, Obs>                       |
|  Pure numerical core. Template specialization fully |
|  resolved at compile time.                          |
+-----------------------------------------------------+
       |                |                  |
       v                v                  v
   Motion trait    Obs trait        Eigen fixed-size
   - N             - M              matrices
   - State alias   - Meas alias     (stack-allocated)
   - predict()     - h(state)
   - process_      - residual()       <- override for theta wrap
     noise()       - measurement_
                    noise()
```

Static polymorphism via templates. Trait methods are inlined at compile time. Eigen `Matrix<double, N, N>` and `Matrix<double, 2*N+1, N>` are stack-allocated.

## File Layout

```text
cpp/include/immtrack/
  ukf.hpp                  # Main UKF template (full unscented transform)
  motion.hpp               # PosVxyzYawCV (replaces CV3D / CTRV3D)
  observations.hpp         # PosYawObs   (replaces BBox3DObs)
  errors.hpp               # CovarianceNotPsd, InvalidArgument, NumericalError
  detail/
    angle.hpp              # wrap_angle helper
    sigma_points.hpp       # Merwe scaled sigma point generator + LLT/LDLT fallback

bindings/
  _core.cc                 # Pybind11 module — single UkfPosVxyzYawCV class

src/immtrack/
  __init__.py              # Re-exports UkfPosVxyzYawCV + exceptions
  main.py                  # Untouched

tests/cpp/                 # NEW
  CMakeLists.txt           # Catch2 v3 via FetchContent
  helpers/
    eigen_matchers.hpp     # IsApprox / IsPsd custom matchers
    fixtures.hpp           # Shared priors
  test_angle.cpp
  test_sigma_points.cpp
  test_ukf_predict.cpp
  test_ukf_update.cpp
  test_traits.cpp

tests/python/
  test_smoke.py            # Rewritten — binding sanity, dimension validation,
                           #   exception propagation. Not duplicating C++ tests.

CMakeLists.txt             # Adds INTERFACE target immtrack_core,
                           #   plus optional tests/cpp subdir guarded by
                           #   BUILD_TESTING and PROJECT_IS_TOP_LEVEL.

docs/superpowers/specs/
  2026-05-04-immtrack-initial-ukf-design.md   # This file
```

## Naming Convention

State / motion structs follow `<Content><Model>`:

| Token | Meaning                 | Components       |
| ----- | ----------------------- | ---------------- |
| Pos   | position                | x, y, z          |
| Vxyz  | Cartesian velocity (3D) | vx, vy, vz       |
| Vxy   | Cartesian velocity (2D) | vx, vy           |
| Yaw   | yaw angle               | θ                |
| Wz    | yaw rate around z       | ω                |
| Axyz  | Cartesian acceleration  | ax, ay, az       |
| Spd   | scalar speed            | v                |
| Size  | dimensions              | l, w, h          |
| BBox  | full 3D box             | Pos + Size + Yaw |

Motion suffixes: `CV`, `CTRV`, `CA`, `CTRA`, `Bicycle`.

Observation structs end with `Obs`. They list only what is measured (no model suffix).

This spec uses:

- Motion: `PosVxyzYawCV` (state `[x, y, z, vx, vy, vz, θ]`, N = 7)
- Observation: `PosYawObs` (measurement `[x, y, z, θ]`, M = 4)
- Python class: `UkfPosVxyzYawCV`

## Trait Contract

### Motion trait

```cpp
struct PosVxyzYawCV {
  static constexpr int N = 7;
  using State = Eigen::Matrix<double, N, 1>;
  using Cov   = Eigen::Matrix<double, N, N>;

  // Required
  static State predict(const State& x, double dt);
  static Cov   process_noise(double dt);

  // Required when state contains angular dimensions
  static State weighted_mean(
      const Eigen::Matrix<double, N, 2*N+1>& sigma_points,
      const Eigen::Matrix<double, 2*N+1, 1>& weights);

  static State residual(const State& a, const State& b);  // a − b with theta wrap
};
```

### Observation trait

```cpp
struct PosYawObs {
  static constexpr int M = 4;
  using Meas  = Eigen::Matrix<double, M, 1>;
  using Noise = Eigen::Matrix<double, M, M>;

  // Required
  template <class State>
  static Meas h(const State& x);
  static Noise measurement_noise();

  // Required when measurement contains angular dimensions
  static Meas weighted_mean(
      const auto& sigma_points_in_meas_space,
      const auto& weights);
  static Meas residual(const Meas& a, const Meas& b);
};
```

### `if constexpr` detection

UKF detects whether a trait provides `weighted_mean` / `residual` via C++20 `requires` and falls back to a default weighted average / element-wise subtraction when absent. For `PosVxyzYawCV` and `PosYawObs`, both overrides MUST be implemented because their angular dimensions need wrapping.

## UKF Numerical Algorithm

### Sigma points (Merwe scaled)

```text
λ = α² · (N + κ) − N
χ_0 = μ
χ_i     = μ + (√((N + λ) · Σ))_i        for i = 1..N
χ_(N+i) = μ − (√((N + λ) · Σ))_i        for i = 1..N
```

Square root via Eigen `LLT` (lower triangular). On `LLT` failure, fall back to `LDLT` with negative eigenvalues clamped to zero. If `LDLT` also fails, throw `CovarianceNotPsd`.

Weights:

```text
W_m_0 = λ / (N + λ)
W_c_0 = λ / (N + λ) + (1 − α² + β)
W_m_i = W_c_i = 1 / (2 (N + λ))    for i = 1..2N
```

Defaults: `α = 1e−3`, `β = 2`, `κ = 0`. Configurable via UKF constructor.

### Predict

```text
Inputs: μ, Σ, dt
Outputs: μ', Σ'

1. Generate sigma points {χ_i} from (μ, Σ).
2. χ_i' = Motion::predict(χ_i, dt).
3. μ' = Motion::weighted_mean({χ_i'}, W_m).
4. Σ' = Σ_i W_c_i · (Motion::residual(χ_i', μ')) · (Motion::residual(χ_i', μ'))ᵀ
        + Motion::process_noise(dt).
5. Symmetrize: Σ' ← 0.5 (Σ' + Σ'ᵀ).
```

### Update

```text
Inputs: μ, Σ, measurement z
Outputs: μ', Σ', NIS (scalar)

1. Generate sigma points {χ_i} from (μ, Σ).      # use latest covariance
2. ζ_i = Obs::h(χ_i).
3. ẑ   = Obs::weighted_mean({ζ_i}, W_m).
4. S   = Σ_i W_c_i · (Obs::residual(ζ_i, ẑ)) · (Obs::residual(ζ_i, ẑ))ᵀ
        + Obs::measurement_noise().
5. T   = Σ_i W_c_i · (Motion::residual(χ_i, μ)) · (Obs::residual(ζ_i, ẑ))ᵀ.
6. K   = T · S⁻¹                                  # via LDLT solve, no inverse
7. r   = Obs::residual(z, ẑ).
8. μ'  = μ + K r.
9. Σ'  = Σ − K S Kᵀ.
10. Symmetrize Σ'.
11. NIS = rᵀ S⁻¹ r.
```

`NIS` is returned by `update()` for use by future IMM and gating logic. Scalar return is essentially free.

## Error Handling

```cpp
namespace immtrack {
class CovarianceNotPsd : public std::runtime_error { ... };
class InvalidArgument  : public std::invalid_argument { ... };
class NumericalError   : public std::runtime_error { ... };
}
```

| Condition                                | Exception          |
| ---------------------------------------- | ------------------ |
| LLT and LDLT both fail (Σ or S not PSD)  | `CovarianceNotPsd` |
| `dt ≤ 0`, init with non-square cov, etc. | `InvalidArgument`  |
| Any other numerical breakdown            | `NumericalError`   |

Pybind11 maps these via `py::register_exception` (with `InvalidArgument` aliased to Python `ValueError`).

`predict()` and `update()` do **not** check for prior `init()` — the default constructor leaves μ = 0, Σ = I, which is a usable identity prior. Documented in headers.

NaN inputs are not actively detected at runtime. The framework trusts callers; debug builds may add `assert` later if needed.

### Numerical hygiene

- After every `predict` and `update`: `Σ ← 0.5 (Σ + Σᵀ)` to suppress drift.
- LLT fails → LDLT fallback with `D ← max(D, 0)`.
- Angle wrap via `atan2(sin(a), cos(a))`, not `fmod`.

## Constructor / API

```cpp
template <class Motion, class Obs>
class UKF {
 public:
  static constexpr int N = Motion::N;
  static constexpr int M = Obs::M;
  using StateVec = Eigen::Matrix<double, N, 1>;
  using StateMat = Eigen::Matrix<double, N, N>;
  using MeasVec  = Eigen::Matrix<double, M, 1>;

  UKF();
  UKF(double alpha, double beta, double kappa);

  void init(const StateVec& x, const StateMat& P);
  void predict(double dt);
  double update(const MeasVec& z);   // returns NIS

  const StateVec& state() const noexcept;
  const StateMat& covariance() const noexcept;

 private:
  StateVec x_;
  StateMat P_;
  double alpha_, beta_, kappa_;
  // Cached weights computed from (α, β, κ, N) at construction.
};
```

## Python API

```python
from immtrack import (
    UkfPosVxyzYawCV,
    CovarianceNotPsd,
    InvalidArgument,
    NumericalError,
)

ukf = UkfPosVxyzYawCV()                       # defaults
# or UkfPosVxyzYawCV(alpha=1e-3, beta=2.0, kappa=0.0)

ukf.init(state=np.zeros(7), cov=np.eye(7))
ukf.predict(dt=0.1)
nis: float = ukf.update(measurement=np.zeros(4))

ukf.state         # ndarray shape (7,)
ukf.covariance    # ndarray shape (7, 7)
UkfPosVxyzYawCV.N # 7   (class-level constants)
UkfPosVxyzYawCV.M # 4
```

## Testing Strategy

### C++ (Catch2 v3 via CMake `FetchContent`)

| File                    | Coverage                                                                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_angle.cpp`        | `wrap_angle`: in-range, ±π boundary, multi-revolution, NaN/Inf behaviour.                                                                        |
| `test_sigma_points.cpp` | Reconstruction of (μ, Σ) from generated points (≤ 1e−10 error); symmetry; multiple N via `TEMPLATE_TEST_CASE`; LLT-fail → LDLT path.             |
| `test_ukf_predict.cpp`  | Linear motion matches plain KF; `dt = 0` is no-op; `dt < 0` throws; PSD preserved over 1000 cycles; theta crossing ±π.                           |
| `test_ukf_update.cpp`   | Zero innovation → state unchanged, cov shrinks; theta-wrap residual correctness; convergence under noisy measurements; `NIS ≥ 0`; PSD preserved. |
| `test_traits.cpp`       | `STATIC_REQUIRE` on N, M, type aliases; `requires` checks on trait method presence.                                                              |

Custom matchers in `helpers/eigen_matchers.hpp`:

- `IsApprox(Eigen::MatrixXd expected, double tol)`
- `IsPsd()` (LDLT with positive-D check)

### Python (`tests/python/test_smoke.py`)

Pytest-based binding-level checks only. Does not duplicate C++ numerical tests.

- Default construction shapes.
- `init` → `predict` → `update` smoke run.
- Measurement dimension mismatch raises `ValueError` / `TypeError`.
- Pass non-PSD covariance → `immtrack.CovarianceNotPsd` raised on first `predict()`.

### CMake test wiring

```cmake
option(IMMTRACK_BUILD_TESTS "Build C++ unit tests" OFF)
if(IMMTRACK_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
```

Default `OFF` ensures `pip install` / `uv build` (scikit-build-core invokes CMake without `-DIMMTRACK_BUILD_TESTS=ON`) does not download or compile Catch2. Developers opt in explicitly: `cmake -B build -DIMMTRACK_BUILD_TESTS=ON`.

## Build / CMake Changes

```cmake
cmake_minimum_required(VERSION 3.15)
project(immtrack LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.4 CONFIG REQUIRED)

# Header-only INTERFACE library — usable by both the pybind module and tests.
add_library(immtrack_core INTERFACE)
target_include_directories(immtrack_core INTERFACE cpp/include)
target_link_libraries(immtrack_core INTERFACE Eigen3::Eigen)
target_compile_features(immtrack_core INTERFACE cxx_std_20)

pybind11_add_module(_core bindings/_core.cc)
target_link_libraries(_core PRIVATE immtrack_core)
install(TARGETS _core DESTINATION immtrack)

option(IMMTRACK_BUILD_TESTS "Build C++ unit tests" OFF)
if(IMMTRACK_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
```

## Migration Notes

- Delete `CV3D`, `CTRV3D` from `cpp/include/immtrack/motion.hpp`.
- Delete `BBox3DObs` from `cpp/include/immtrack/observations.hpp`.
- Rewrite `bindings/_core.cc` to expose only `UkfPosVxyzYawCV` + exception types.
- Rewrite `src/immtrack/__init__.py` and `tests/python/test_smoke.py` to match the new dimensions and class name.
- Update header comment in `ukf.hpp` to describe trait contract precisely.

## Extension Recipe

Adding a new `UKF<Motion, Obs>` instance:

1. Add the new motion (or obs) struct to `motion.hpp` / `observations.hpp` following the trait contract. Provide `weighted_mean` and `residual` overrides if any dimension is angular.
2. In `bindings/_core.cc`, add `using Ukf<Name> = UKF<...>;` and `bind_filter<Ukf<Name>>(m, "Ukf<Name>");`.
3. Re-export from `src/immtrack/__init__.py`.
4. Add C++ tests in `tests/cpp/test_traits.cpp` (compile-time shape) and a smoke case in `test_ukf_predict.cpp` if dynamics differ meaningfully.

The UKF template itself is not modified.
