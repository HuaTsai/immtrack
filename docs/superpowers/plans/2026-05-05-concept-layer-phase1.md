# Concept Layer + 8D State (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the 4-axis concept design (StateSpace / MotionModel / ObservationModel / Filter) from `ref/concept_kf_demo.hpp` into the production headers, and grow the existing 7D state (`x,y,z,vx,vy,vz,yaw`) by one dimension to 8D (`+ yaw_rate`) so that Phase 2's IMM with CTRV has a place to write yaw rate. No IMM in this phase.

**Architecture:** Header-only C++20 library. New files: `concepts.hpp`, `detail/euclidean.hpp`, `state_spaces.hpp`. The `EuclideanWithAngles<Dim, AngleIdx...>` template generates StateSpace types with angle-aware `boxplus`/`boxminus`/`weighted_mean`, eliminating per-state-space hand-coding. Existing `PosVxyzYawCV`, `PosYawObs`, `UKF`, `Track`, `BBoxTracker` are refactored in-place to satisfy the new concepts; numeric equivalence with the existing tracker is the success criterion.

**Tech Stack:** C++20 (concepts, fold expressions). Eigen 3.4 fixed-size matrices. Catch2 v3 for tests. CMake (`-DIMMTRACK_BUILD_TESTS=ON`). pybind11 for Python bindings. pytest for Python tests.

**Spec reference:** `docs/superpowers/specs/2026-05-05-tracker-architecture-redesign.md`

---

## File Structure

| File                                        | Action | Responsibility                                                                                                                          |
| ------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `cpp/include/immtrack/concepts.hpp`         | Create | `StateSpace`, `MotionModel`, `HasMotionJacobian`, `LinearMotion`, `ObservationModel`, `HasObsJacobian`, `LinearObs`, `Filter` concepts. |
| `cpp/include/immtrack/detail/euclidean.hpp` | Create | `EuclideanWithAngles<Dim, AngleIdx...>` providing `boxplus`, `boxminus`, `weighted_mean`.                                               |
| `cpp/include/immtrack/state_spaces.hpp`     | Create | `XYZVxVyVzYawYawRateSpace` (8D, yaw at index 6, yaw_rate at index 7).                                                                   |
| `cpp/include/immtrack/motion.hpp`           | Modify | `PosVxyzYawCV` gains `StateSpace` typedef, grows to 8D; predict leaves `yaw_rate` unchanged; process_noise zero on yaw_rate row/col.    |
| `cpp/include/immtrack/observations.hpp`     | Modify | `PosYawObs` gains `StateSpace` and `MeasSpace` typedefs.                                                                                |
| `cpp/include/immtrack/ukf.hpp`              | Modify | Concept-constrained on Motion/Obs; `static_assert(Filter<UKF<PosVxyzYawCV, PosYawObs>>)`.                                               |
| `cpp/include/immtrack/tracker.hpp`          | Modify | `Track::Track(...)` — 8D init (yaw_rate=0, P0(7,7)=small variance).                                                                     |
| `bindings/_core.cc`                         | Modify | No code change expected (templated on `Filter`); verify `Filter::N` propagates as 8.                                                    |
| `tests/cpp/test_concepts.cc`                | Create | static_assert tests for each concept against stub types.                                                                                |
| `tests/cpp/test_euclidean.cc`               | Create | `EuclideanWithAngles` boxplus/boxminus/weighted_mean unit tests.                                                                        |
| `tests/cpp/test_state_spaces.cc`            | Create | `XYZVxVyVzYawYawRateSpace` static_assert + accessor enum tests.                                                                         |
| `tests/cpp/test_traits.cc`                  | Modify | Update N from 7 to 8; add `StateSpace`/`MeasSpace` typedef checks.                                                                      |
| `tests/cpp/test_ukf_predict.cc`             | Modify | Update fixtures to 8D state vectors.                                                                                                    |
| `tests/cpp/test_ukf_update.cc`              | Modify | Same.                                                                                                                                   |
| `tests/cpp/test_ukf_predict_measurement.cc` | Modify | Same.                                                                                                                                   |
| `tests/cpp/test_bbox_tracker.cc`            | Modify | Same; verify Track init handles 8D.                                                                                                     |
| `tests/cpp/CMakeLists.txt`                  | Modify | Register `test_concepts`, `test_euclidean`, `test_state_spaces`.                                                                        |

---

## Build & Test Commands (used throughout)

Configure once:

```bash
cmake -S . -B build-test -DIMMTRACK_BUILD_TESTS=ON
```

Build all tests:

```bash
cmake --build build-test -j
```

Run all C++ tests:

```bash
ctest --test-dir build-test --output-on-failure
```

Run a single test executable:

```bash
./build-test/tests/cpp/test_concepts
```

Run Python tests:

```bash
pytest tests/python/
```

Run AMOTA equivalence smoke (Phase 1 acceptance):

```bash
pytest tests/python/test_amota.py -v -k "tracker"
```

---

## Task 1: Concepts header (`concepts.hpp`)

**Files:**

- Create: `cpp/include/immtrack/concepts.hpp`
- Test: `tests/cpp/test_concepts.cc`
- Modify: `tests/cpp/CMakeLists.txt`

### Step 1.1: Write failing test `test_concepts.cc`

- [ ] Create `tests/cpp/test_concepts.cc`:

```cpp
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <immtrack/concepts.hpp>

namespace {

// Minimal stub satisfying StateSpace.
struct StubSpace {
  using Scalar  = double;
  using State   = Eigen::Matrix<double, 3, 1>;
  using Tangent = Eigen::Matrix<double, 3, 1>;
  using Cov     = Eigen::Matrix<double, 3, 3>;
  static constexpr int state_dim   = 3;
  static constexpr int tangent_dim = 3;
  static State boxplus(const State& x, const Tangent& dx) noexcept { return x + dx; }
  static Tangent boxminus(const State& a, const State& b) noexcept { return a - b; }
};

// Minimal motion (linear) over StubSpace.
struct StubMotion {
  using StateSpace = StubSpace;
  static StateSpace::State predict(const StateSpace::State& x, double) noexcept { return x; }
  static StateSpace::Cov   process_noise(double) noexcept { return StateSpace::Cov::Identity(); }
  static StateSpace::Cov   F_jacobian(const StateSpace::State&, double) noexcept {
    return StateSpace::Cov::Identity();
  }
  static StateSpace::Cov   F_matrix(double) noexcept { return StateSpace::Cov::Identity(); }
};

// 2D measurement space.
struct StubMeasSpace {
  using Scalar  = double;
  using State   = Eigen::Matrix<double, 2, 1>;
  using Tangent = Eigen::Matrix<double, 2, 1>;
  using Cov     = Eigen::Matrix<double, 2, 2>;
  static constexpr int state_dim   = 2;
  static constexpr int tangent_dim = 2;
  static State   boxplus (const State& x, const Tangent& dx) noexcept { return x + dx; }
  static Tangent boxminus(const State& a, const State& b) noexcept { return a - b; }
};

// Minimal obs (linear) over StubSpace -> StubMeasSpace.
struct StubObs {
  using StateSpace = StubSpace;
  using MeasSpace  = StubMeasSpace;
  using HMat = Eigen::Matrix<double, 2, 3>;
  static MeasSpace::State h(const StateSpace::State& x) noexcept { return x.head<2>(); }
  static MeasSpace::Cov   measurement_noise() noexcept { return MeasSpace::Cov::Identity(); }
  static HMat             H_jacobian(const StateSpace::State&) noexcept { return H_matrix(); }
  static HMat             H_matrix() noexcept {
    HMat H = HMat::Zero(); H(0,0) = 1; H(1,1) = 1; return H;
  }
};

}  // namespace

TEST_CASE("StateSpace concept", "[concepts]") {
  STATIC_REQUIRE(immtrack::StateSpace<StubSpace>);
  STATIC_REQUIRE(immtrack::StateSpace<StubMeasSpace>);
}

TEST_CASE("MotionModel refinement", "[concepts]") {
  STATIC_REQUIRE(immtrack::MotionModel<StubMotion>);
  STATIC_REQUIRE(immtrack::HasMotionJacobian<StubMotion>);
  STATIC_REQUIRE(immtrack::LinearMotion<StubMotion>);
}

TEST_CASE("ObservationModel refinement", "[concepts]") {
  STATIC_REQUIRE(immtrack::ObservationModel<StubObs>);
  STATIC_REQUIRE(immtrack::HasObsJacobian<StubObs>);
  STATIC_REQUIRE(immtrack::LinearObs<StubObs>);
}
```

- [ ] Add the test to `tests/cpp/CMakeLists.txt` after the existing list:

```cmake
immtrack_add_test(test_concepts)
```

### Step 1.2: Run test to verify it fails

- [ ] Run:

```bash
cmake --build build-test -j 2>&1 | head -40
```

Expected: build error mentioning `immtrack/concepts.hpp` not found.

### Step 1.3: Write `concepts.hpp`

- [ ] Create `cpp/include/immtrack/concepts.hpp`:

```cpp
#pragma once

#include <Eigen/Core>
#include <concepts>
#include <type_traits>

namespace immtrack {

// 1. StateSpace — geometry of the state manifold.
template <class S>
concept StateSpace = requires(typename S::State   x,
                              typename S::Tangent dx,
                              typename S::State   a,
                              typename S::State   b) {
  typename S::Scalar;
  typename S::State;
  typename S::Tangent;
  typename S::Cov;
  { S::state_dim   } -> std::convertible_to<int>;
  { S::tangent_dim } -> std::convertible_to<int>;
  requires std::floating_point<typename S::Scalar>;
  { S::boxplus(x, dx) } -> std::same_as<typename S::State>;
  { S::boxminus(a, b) } -> std::same_as<typename S::Tangent>;
};

// 2. MotionModel — dynamics over a StateSpace.
template <class M>
concept MotionModel =
    requires(const typename M::StateSpace::State& x,
             typename M::StateSpace::Scalar       dt) {
      requires StateSpace<typename M::StateSpace>;
      { M::predict(x, dt) }    -> std::same_as<typename M::StateSpace::State>;
      { M::process_noise(dt) } -> std::same_as<typename M::StateSpace::Cov>;
    };

template <class M>
concept HasMotionJacobian =
    MotionModel<M> &&
    requires(const typename M::StateSpace::State& x,
             typename M::StateSpace::Scalar       dt) {
      { M::F_jacobian(x, dt) } -> std::same_as<typename M::StateSpace::Cov>;
    };

template <class M>
concept LinearMotion =
    HasMotionJacobian<M> &&
    requires(typename M::StateSpace::Scalar dt) {
      { M::F_matrix(dt) } -> std::same_as<typename M::StateSpace::Cov>;
    };

// 3. ObservationModel — projection from StateSpace into MeasSpace.
template <class O>
using ObsJacMat = Eigen::Matrix<typename O::StateSpace::Scalar,
                                O::MeasSpace::tangent_dim,
                                O::StateSpace::tangent_dim>;

template <class O>
concept ObservationModel =
    requires(const typename O::StateSpace::State& x) {
      requires StateSpace<typename O::StateSpace>;
      requires StateSpace<typename O::MeasSpace>;
      requires std::same_as<typename O::StateSpace::Scalar,
                            typename O::MeasSpace::Scalar>;
      { O::h(x) } -> std::same_as<typename O::MeasSpace::State>;
      { O::measurement_noise() } -> std::same_as<typename O::MeasSpace::Cov>;
    };

template <class O>
concept HasObsJacobian =
    ObservationModel<O> &&
    requires(const typename O::StateSpace::State& x) {
      { O::H_jacobian(x) } -> std::same_as<ObsJacMat<O>>;
    };

template <class O>
concept LinearObs =
    HasObsJacobian<O> &&
    requires {
      { O::H_matrix() } -> std::same_as<ObsJacMat<O>>;
    };

// 4. Filter — uniform external API for KF / EKF / UKF / IMM.
template <class F>
concept Filter =
    requires { typename F::StateSpace; typename F::MeasSpace; } &&
    StateSpace<typename F::StateSpace> &&
    StateSpace<typename F::MeasSpace>  &&
    requires(F& f, double dt,
             const typename F::MeasSpace::State& z,
             const F& cf) {
      { f.predict(dt) } -> std::same_as<void>;
      { f.update(z)   } -> std::convertible_to<double>;
      { cf.state()      } -> std::convertible_to<const typename F::StateSpace::State&>;
      { cf.covariance() } -> std::convertible_to<const typename F::StateSpace::Cov&>;
    };

}  // namespace immtrack
```

### Step 1.4: Run test to verify it passes

- [ ] Run:

```bash
cmake --build build-test -j --target test_concepts && ./build-test/tests/cpp/test_concepts
```

Expected: `All tests passed (3 assertions in 3 test cases)`.

### Step 1.5: Commit

- [ ] Run:

```bash
git add cpp/include/immtrack/concepts.hpp tests/cpp/test_concepts.cc tests/cpp/CMakeLists.txt
git commit -m "feat(concepts): add StateSpace/MotionModel/ObservationModel/Filter concepts"
```

---

## Task 2: `EuclideanWithAngles<Dim, AngleIdx...>` template

**Files:**

- Create: `cpp/include/immtrack/detail/euclidean.hpp`
- Test: `tests/cpp/test_euclidean.cc`
- Modify: `tests/cpp/CMakeLists.txt`

### Step 2.1: Write failing test `test_euclidean.cc`

- [ ] Create `tests/cpp/test_euclidean.cc`:

```cpp
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <immtrack/detail/euclidean.hpp>
#include <numbers>

using immtrack::detail::EuclideanWithAngles;

TEST_CASE("EuclideanWithAngles<3>: pure Euclidean (no angle indices)", "[euclidean]") {
  using Space = EuclideanWithAngles<3>;
  STATIC_REQUIRE(Space::state_dim   == 3);
  STATIC_REQUIRE(Space::tangent_dim == 3);

  Space::State a; a << 1.0, 2.0, 3.0;
  Space::Tangent dx; dx << 0.5, -0.5, 1.0;
  const auto b = Space::boxplus(a, dx);
  REQUIRE(b(0) == Catch::Approx(1.5));
  REQUIRE(b(1) == Catch::Approx(1.5));
  REQUIRE(b(2) == Catch::Approx(4.0));
  const auto d = Space::boxminus(b, a);
  REQUIRE(d.isApprox(dx));
}

TEST_CASE("EuclideanWithAngles<8, 6>: angle wrap on idx 6 only", "[euclidean]") {
  using Space = EuclideanWithAngles<8, 6>;
  Space::State a = Space::State::Zero();
  a(6) = std::numbers::pi - 0.05;
  Space::Tangent dx = Space::Tangent::Zero();
  dx(6) = 0.10;
  const auto b = Space::boxplus(a, dx);
  // (pi - 0.05) + 0.10 = pi + 0.05  -> wraps to -pi + 0.05
  REQUIRE(b(6) == Catch::Approx(-std::numbers::pi + 0.05).margin(1e-12));
  // Other dimensions untouched.
  for (int i = 0; i < 8; ++i) {
    if (i == 6) continue;
    REQUIRE(b(i) == 0.0);
  }
}

TEST_CASE("EuclideanWithAngles<8, 6>: boxminus wraps yaw difference", "[euclidean]") {
  using Space = EuclideanWithAngles<8, 6>;
  Space::State a = Space::State::Zero();
  Space::State b = Space::State::Zero();
  a(6) =  std::numbers::pi - 0.10;
  b(6) = -std::numbers::pi + 0.10;
  // Naive a - b = 2*pi - 0.20  -> wraps to -0.20
  const auto d = Space::boxminus(a, b);
  REQUIRE(d(6) == Catch::Approx(-0.20).margin(1e-12));
}

TEST_CASE("EuclideanWithAngles<8, 6>: weighted_mean averages non-angles linearly", "[euclidean]") {
  using Space = EuclideanWithAngles<8, 6>;
  constexpr int K = 3;
  Eigen::Matrix<double, 8, K> sigmas;
  sigmas.setZero();
  sigmas(0, 0) = 1.0; sigmas(0, 1) = 2.0; sigmas(0, 2) = 3.0;
  Eigen::Matrix<double, K, 1> w; w << 0.5, 0.25, 0.25;
  const auto mean = Space::template weighted_mean<K>(sigmas, w);
  REQUIRE(mean(0) == Catch::Approx(0.5*1.0 + 0.25*2.0 + 0.25*3.0));
}

TEST_CASE("EuclideanWithAngles<8, 6>: weighted_mean handles yaw across pi", "[euclidean]") {
  using Space = EuclideanWithAngles<8, 6>;
  constexpr int K = 2;
  Eigen::Matrix<double, 8, K> sigmas;
  sigmas.setZero();
  sigmas(6, 0) =  std::numbers::pi - 0.05;
  sigmas(6, 1) = -std::numbers::pi + 0.05;
  Eigen::Matrix<double, K, 1> w; w << 0.5, 0.5;
  const auto mean = Space::template weighted_mean<K>(sigmas, w);
  // Intrinsic mean of (pi - 0.05) and (-pi + 0.05) is +/- pi (wrap point).
  // Result must be wrapped to (-pi, pi]. We accept either +pi or -pi numerically.
  REQUIRE(std::abs(std::abs(mean(6)) - std::numbers::pi) < 1e-10);
}
```

- [ ] Add to `tests/cpp/CMakeLists.txt`:

```cmake
immtrack_add_test(test_euclidean)
```

### Step 2.2: Run test to verify it fails

- [ ] Run:

```bash
cmake --build build-test -j --target test_euclidean 2>&1 | head -20
```

Expected: build error — `immtrack/detail/euclidean.hpp` not found.

### Step 2.3: Implement `detail/euclidean.hpp`

- [ ] Create `cpp/include/immtrack/detail/euclidean.hpp`:

```cpp
#pragma once

#include <Eigen/Core>
#include <immtrack/detail/angle.hpp>

namespace immtrack::detail {

// Euclidean state space with angle-wrapped indices.
//
// All non-angle dimensions are plain Euclidean. Each index listed in
// `AngleIdx` lives on S^1 (theta in (-pi, pi]) and uses `wrap_angle`
// for boxplus / boxminus / weighted_mean.
//
// Used as the implementation backbone of named StateSpaces in
// state_spaces.hpp (e.g. XYZVxVyVzYawYawRateSpace = EuclideanWithAngles<8, 6>).
template <int Dim, int... AngleIdx>
struct EuclideanWithAngles {
  using Scalar  = double;
  using State   = Eigen::Matrix<double, Dim, 1>;
  using Tangent = Eigen::Matrix<double, Dim, 1>;
  using Cov     = Eigen::Matrix<double, Dim, Dim>;
  static constexpr int state_dim   = Dim;
  static constexpr int tangent_dim = Dim;

  static State boxplus(const State& x, const Tangent& dx) noexcept {
    State r = x + dx;
    ((r(AngleIdx) = wrap_angle(r(AngleIdx))), ...);
    return r;
  }

  static Tangent boxminus(const State& a, const State& b) noexcept {
    Tangent r = a - b;
    ((r(AngleIdx) = wrap_angle(r(AngleIdx))), ...);
    return r;
  }

  // Weighted mean on the manifold:
  //   - non-angle dimensions: ordinary weighted Euclidean mean.
  //   - each angle dimension: intrinsic mean using wrap_angle deltas
  //     anchored at sigmas.col(0)'s value (numerically stable for
  //     small spreads typical of UKF sigma points).
  template <int K>
  static State weighted_mean(const Eigen::Matrix<double, Dim, K>& sigmas,
                              const Eigen::Matrix<double, K, 1>& weights) noexcept {
    State mean = State::Zero();
    // Plain weighted sum over all dims.
    for (int i = 0; i < K; ++i) {
      mean.noalias() += weights(i) * sigmas.col(i);
    }
    // Recompute angle dims as intrinsic mean (overwrites the linear
    // sum above for those indices).
    ((mean(AngleIdx) = intrinsic_angle_mean_<K>(sigmas, weights, AngleIdx)), ...);
    return mean;
  }

 private:
  template <int K>
  static double intrinsic_angle_mean_(
      const Eigen::Matrix<double, Dim, K>& sigmas,
      const Eigen::Matrix<double, K, 1>& weights,
      int idx) noexcept {
    const double anchor = sigmas(idx, 0);
    double delta = 0.0;
    for (int i = 0; i < K; ++i) {
      delta += weights(i) * wrap_angle(sigmas(idx, i) - anchor);
    }
    return wrap_angle(anchor + delta);
  }
};

}  // namespace immtrack::detail
```

### Step 2.4: Run test to verify it passes

- [ ] Run:

```bash
cmake --build build-test -j --target test_euclidean && ./build-test/tests/cpp/test_euclidean
```

Expected: `All tests passed`.

### Step 2.5: Commit

- [ ] Run:

```bash
git add cpp/include/immtrack/detail/euclidean.hpp tests/cpp/test_euclidean.cc tests/cpp/CMakeLists.txt
git commit -m "feat(detail): add EuclideanWithAngles state-space generator"
```

---

## Task 3: `XYZVxVyVzYawYawRateSpace` named state space

**Files:**

- Create: `cpp/include/immtrack/state_spaces.hpp`
- Test: `tests/cpp/test_state_spaces.cc`
- Modify: `tests/cpp/CMakeLists.txt`

### Step 3.1: Write failing test `test_state_spaces.cc`

- [ ] Create `tests/cpp/test_state_spaces.cc`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <immtrack/concepts.hpp>
#include <immtrack/state_spaces.hpp>
#include <type_traits>

using immtrack::XYZVxVyVzYawYawRateSpace;

TEST_CASE("XYZVxVyVzYawYawRateSpace satisfies StateSpace", "[state_spaces]") {
  STATIC_REQUIRE(immtrack::StateSpace<XYZVxVyVzYawYawRateSpace>);
  STATIC_REQUIRE(XYZVxVyVzYawYawRateSpace::state_dim   == 8);
  STATIC_REQUIRE(XYZVxVyVzYawYawRateSpace::tangent_dim == 8);
}

TEST_CASE("XYZVxVyVzYawYawRateSpace named indices", "[state_spaces]") {
  using S = XYZVxVyVzYawYawRateSpace;
  STATIC_REQUIRE(S::X        == 0);
  STATIC_REQUIRE(S::Y        == 1);
  STATIC_REQUIRE(S::Z        == 2);
  STATIC_REQUIRE(S::VX       == 3);
  STATIC_REQUIRE(S::VY       == 4);
  STATIC_REQUIRE(S::VZ       == 5);
  STATIC_REQUIRE(S::YAW      == 6);
  STATIC_REQUIRE(S::YAW_RATE == 7);
}
```

- [ ] Add to `tests/cpp/CMakeLists.txt`:

```cmake
immtrack_add_test(test_state_spaces)
```

### Step 3.2: Run test to verify it fails

- [ ] Run:

```bash
cmake --build build-test -j --target test_state_spaces 2>&1 | head -10
```

Expected: build error — `immtrack/state_spaces.hpp` not found.

### Step 3.3: Implement `state_spaces.hpp`

- [ ] Create `cpp/include/immtrack/state_spaces.hpp`:

```cpp
#pragma once

#include <immtrack/detail/euclidean.hpp>

namespace immtrack {

// 8D state for ground-vehicle 3D bbox tracking.
// Index layout: [x, y, z, vx, vy, vz, yaw, yaw_rate].
// `yaw` (idx 6) wraps to (-pi, pi]; `yaw_rate` is plain Euclidean.
struct XYZVxVyVzYawYawRateSpace : detail::EuclideanWithAngles<8, /*yaw=*/6> {
  enum : int {
    X        = 0,
    Y        = 1,
    Z        = 2,
    VX       = 3,
    VY       = 4,
    VZ       = 5,
    YAW      = 6,
    YAW_RATE = 7,
  };
};

}  // namespace immtrack
```

### Step 3.4: Run test to verify it passes

- [ ] Run:

```bash
cmake --build build-test -j --target test_state_spaces && ./build-test/tests/cpp/test_state_spaces
```

Expected: `All tests passed`.

### Step 3.5: Commit

- [ ] Run:

```bash
git add cpp/include/immtrack/state_spaces.hpp tests/cpp/test_state_spaces.cc tests/cpp/CMakeLists.txt
git commit -m "feat(state_spaces): add XYZVxVyVzYawYawRateSpace (8D, yaw idx=6)"
```

---

## Task 4: Refactor `motion.hpp` to 8D + StateSpace typedef

**Files:**

- Modify: `cpp/include/immtrack/motion.hpp`
- Modify: `tests/cpp/test_traits.cc`

### Step 4.1: Update test_traits.cc to expect 8D

- [ ] Replace the `PosVxyzYawCV` block in `tests/cpp/test_traits.cc` with:

```cpp
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <immtrack/concepts.hpp>
#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <immtrack/state_spaces.hpp>

using immtrack::PosVxyzYawCV;
using immtrack::PosYawObs;
using immtrack::XYZVxVyVzYawYawRateSpace;

TEST_CASE("PosVxyzYawCV: shape and StateSpace typedef", "[traits]") {
  STATIC_REQUIRE(immtrack::MotionModel<PosVxyzYawCV>);
  STATIC_REQUIRE(immtrack::HasMotionJacobian<PosVxyzYawCV>);
  STATIC_REQUIRE(immtrack::LinearMotion<PosVxyzYawCV>);
  STATIC_REQUIRE(std::is_same_v<PosVxyzYawCV::StateSpace, XYZVxVyVzYawYawRateSpace>);
  STATIC_REQUIRE(PosVxyzYawCV::N == 8);
  STATIC_REQUIRE(std::is_same_v<PosVxyzYawCV::State, Eigen::Matrix<double, 8, 1>>);
  STATIC_REQUIRE(std::is_same_v<PosVxyzYawCV::Cov,   Eigen::Matrix<double, 8, 8>>);
}

TEST_CASE("PosVxyzYawCV::predict propagates position, leaves yaw and yaw_rate", "[traits]") {
  PosVxyzYawCV::State x = PosVxyzYawCV::State::Zero();
  x << 1.0, 2.0, 3.0, 0.5, -0.5, 0.0, 0.7, 0.3;
  const auto next = PosVxyzYawCV::predict(x, 0.5);
  REQUIRE(next(0) == Catch::Approx(1.0 + 0.5 * 0.5));    // x + vx*dt
  REQUIRE(next(1) == Catch::Approx(2.0 - 0.5 * 0.5));    // y + vy*dt
  REQUIRE(next(2) == Catch::Approx(3.0));                // z unchanged (vz=0)
  REQUIRE(next(3) == Catch::Approx(0.5));                // vx unchanged
  REQUIRE(next(4) == Catch::Approx(-0.5));               // vy unchanged
  REQUIRE(next(5) == Catch::Approx(0.0));                // vz unchanged
  REQUIRE(next(6) == Catch::Approx(0.7));                // yaw unchanged (passive in CV)
  REQUIRE(next(7) == Catch::Approx(0.3));                // yaw_rate unchanged (passive in CV)
}

TEST_CASE("PosVxyzYawCV: process_noise zero on yaw_rate row/col", "[traits]") {
  const auto Q = PosVxyzYawCV::process_noise(1.0);
  for (int i = 0; i < 8; ++i) {
    REQUIRE(Q(7, i) == Catch::Approx(0.0));
    REQUIRE(Q(i, 7) == Catch::Approx(0.0));
  }
}
```

(Keep the existing `PosYawObs` test cases — they are addressed in Task 5.)

### Step 4.2: Run test to verify it fails

- [ ] Run:

```bash
cmake --build build-test -j --target test_traits 2>&1 | tail -30
```

Expected: build error — `immtrack::PosVxyzYawCV` does not satisfy `MotionModel`, or `StateSpace` typedef missing, or `N` is 7 not 8.

### Step 4.3: Rewrite `motion.hpp`

- [ ] Replace `cpp/include/immtrack/motion.hpp` with:

```cpp
#pragma once

#include <Eigen/Core>
#include <immtrack/state_spaces.hpp>

namespace immtrack {

// Constant-velocity model in 3D Cartesian space + passive yaw + passive yaw_rate.
// state = [x, y, z, vx, vy, vz, yaw, yaw_rate]
//
// CV leaves yaw and yaw_rate unchanged across predict; their covariance grows
// only through process_noise on (x,y,z) (white-noise acceleration in xy/z).
// In Phase 2 a CTRV mode will share the same StateSpace and write yaw/yaw_rate.
struct PosVxyzYawCV {
  using StateSpace = XYZVxVyVzYawYawRateSpace;
  using State = StateSpace::State;
  using Cov   = StateSpace::Cov;
  static constexpr int N = StateSpace::state_dim;

  static State predict(const State& x, double dt) noexcept {
    State next = x;
    next(StateSpace::X) += x(StateSpace::VX) * dt;
    next(StateSpace::Y) += x(StateSpace::VY) * dt;
    next(StateSpace::Z) += x(StateSpace::VZ) * dt;
    return next;
  }

  // White-noise acceleration model on (x,y,z); zero on yaw and yaw_rate so
  // those dimensions stay frozen in CV mode.
  static Cov process_noise(double dt) noexcept {
    Cov Q = Cov::Zero();
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    constexpr double sigma_a2 = 1.0;  // matches existing tuning (Q ~ I*dt for v).
    // Position-position blocks.
    Q(StateSpace::X, StateSpace::X) = (dt3 / 3.0) * sigma_a2;
    Q(StateSpace::Y, StateSpace::Y) = (dt3 / 3.0) * sigma_a2;
    Q(StateSpace::Z, StateSpace::Z) = (dt3 / 3.0) * sigma_a2;
    // Position-velocity cross blocks.
    Q(StateSpace::X, StateSpace::VX) = Q(StateSpace::VX, StateSpace::X) = (dt2 / 2.0) * sigma_a2;
    Q(StateSpace::Y, StateSpace::VY) = Q(StateSpace::VY, StateSpace::Y) = (dt2 / 2.0) * sigma_a2;
    Q(StateSpace::Z, StateSpace::VZ) = Q(StateSpace::VZ, StateSpace::Z) = (dt2 / 2.0) * sigma_a2;
    // Velocity-velocity blocks.
    Q(StateSpace::VX, StateSpace::VX) = dt * sigma_a2;
    Q(StateSpace::VY, StateSpace::VY) = dt * sigma_a2;
    Q(StateSpace::VZ, StateSpace::VZ) = dt * sigma_a2;
    // yaw / yaw_rate rows and columns left at zero.
    return Q;
  }

  // Linear -> jacobian == matrix. Both exposed to satisfy LinearMotion.
  static Cov F_jacobian(const State&, double dt) noexcept { return F_matrix(dt); }
  static Cov F_matrix(double dt) noexcept {
    Cov F = Cov::Identity();
    F(StateSpace::X, StateSpace::VX) = dt;
    F(StateSpace::Y, StateSpace::VY) = dt;
    F(StateSpace::Z, StateSpace::VZ) = dt;
    return F;
  }

  // ===== UKF compatibility shims (delegate to StateSpace) =====
  // The existing UKF reads `Motion::weighted_mean` and `Motion::residual`.
  // Task 6 may switch UKF to call StateSpace directly; until then these
  // shims keep the existing call sites working unchanged.
  template <int K>
  static State weighted_mean(const Eigen::Matrix<double, N, K>& sigmas,
                              const Eigen::Matrix<double, K, 1>& weights) noexcept {
    return StateSpace::template weighted_mean<K>(sigmas, weights);
  }

  static State residual(const State& a, const State& b) noexcept {
    return StateSpace::boxminus(a, b);
  }
};

}  // namespace immtrack
```

### Step 4.4: Run test to verify it passes

- [ ] Run:

```bash
cmake --build build-test -j --target test_traits && ./build-test/tests/cpp/test_traits
```

Expected: `PosVxyzYawCV` cases pass; `PosYawObs` cases may still pass (Task 5 will tighten them).

### Step 4.5: Commit

- [ ] Run:

```bash
git add cpp/include/immtrack/motion.hpp tests/cpp/test_traits.cc
git commit -m "refactor(motion): PosVxyzYawCV grows to 8D, exposes StateSpace typedef"
```

---

## Task 5: Refactor `observations.hpp` to expose `MeasSpace`

**Files:**

- Modify: `cpp/include/immtrack/observations.hpp`
- Modify: `tests/cpp/test_traits.cc`

### Step 5.1: Update test_traits.cc PosYawObs cases

- [ ] Replace the existing `PosYawObs` test cases in `tests/cpp/test_traits.cc` with:

```cpp
TEST_CASE("PosYawObs: shape and concept conformance", "[traits]") {
  STATIC_REQUIRE(immtrack::ObservationModel<PosYawObs>);
  STATIC_REQUIRE(immtrack::HasObsJacobian<PosYawObs>);
  STATIC_REQUIRE(immtrack::LinearObs<PosYawObs>);
  STATIC_REQUIRE(std::is_same_v<PosYawObs::StateSpace, XYZVxVyVzYawYawRateSpace>);
  STATIC_REQUIRE(immtrack::StateSpace<PosYawObs::MeasSpace>);
  STATIC_REQUIRE(PosYawObs::M == 4);
  STATIC_REQUIRE(std::is_same_v<PosYawObs::Meas,  Eigen::Matrix<double, 4, 1>>);
  STATIC_REQUIRE(std::is_same_v<PosYawObs::Noise, Eigen::Matrix<double, 4, 4>>);
}

TEST_CASE("PosYawObs::h projects (x, y, z, yaw) from 8D state", "[traits]") {
  PosVxyzYawCV::State x;
  x << 1.0, 2.0, 3.0, 0.5, -0.5, 0.0, 0.7, 0.3;
  const auto z = PosYawObs::h(x);
  REQUIRE(z(0) == Catch::Approx(1.0));
  REQUIRE(z(1) == Catch::Approx(2.0));
  REQUIRE(z(2) == Catch::Approx(3.0));
  REQUIRE(z(3) == Catch::Approx(0.7));
}
```

### Step 5.2: Run test to verify it fails

- [ ] Run:

```bash
cmake --build build-test -j --target test_traits 2>&1 | tail -20
```

Expected: build error — `PosYawObs::StateSpace` / `PosYawObs::MeasSpace` not found, or `ObservationModel<PosYawObs>` not satisfied.

### Step 5.3: Rewrite `observations.hpp`

- [ ] Replace `cpp/include/immtrack/observations.hpp` with:

```cpp
#pragma once

#include <Eigen/Core>
#include <immtrack/concepts.hpp>
#include <immtrack/detail/euclidean.hpp>
#include <immtrack/state_spaces.hpp>

namespace immtrack {

// Measurement space: [x, y, z, yaw]. Yaw at idx 3 wraps to (-pi, pi].
using PosYawMeasSpace = detail::EuclideanWithAngles<4, /*yaw=*/3>;

// Observation: project (x, y, z, yaw) out of the 8D ground-vehicle state.
struct PosYawObs {
  using StateSpace = XYZVxVyVzYawYawRateSpace;
  using MeasSpace  = PosYawMeasSpace;
  using Meas  = MeasSpace::State;
  using Noise = MeasSpace::Cov;
  using HMat  = ObsJacMat<PosYawObs>;  // 4 x 8
  static constexpr int M = MeasSpace::state_dim;

  static Meas h(const StateSpace::State& x) noexcept {
    Meas z;
    z(0) = x(StateSpace::X);
    z(1) = x(StateSpace::Y);
    z(2) = x(StateSpace::Z);
    z(3) = x(StateSpace::YAW);
    return z;
  }

  static Noise measurement_noise() noexcept { return Noise::Identity(); }

  static HMat H_jacobian(const StateSpace::State&) noexcept { return H_matrix(); }
  static HMat H_matrix() noexcept {
    HMat H = HMat::Zero();
    H(0, StateSpace::X)   = 1.0;
    H(1, StateSpace::Y)   = 1.0;
    H(2, StateSpace::Z)   = 1.0;
    H(3, StateSpace::YAW) = 1.0;
    return H;
  }

  // ===== UKF compatibility shims (delegate to MeasSpace) =====
  template <int K>
  static Meas weighted_mean(const Eigen::Matrix<double, M, K>& sigmas,
                             const Eigen::Matrix<double, K, 1>& weights) noexcept {
    return MeasSpace::template weighted_mean<K>(sigmas, weights);
  }

  static Meas residual(const Meas& a, const Meas& b) noexcept {
    return MeasSpace::boxminus(a, b);
  }
};

}  // namespace immtrack
```

### Step 5.4: Run test to verify it passes

- [ ] Run:

```bash
cmake --build build-test -j --target test_traits && ./build-test/tests/cpp/test_traits
```

Expected: `All tests passed`.

### Step 5.5: Commit

- [ ] Run:

```bash
git add cpp/include/immtrack/observations.hpp tests/cpp/test_traits.cc
git commit -m "refactor(observations): PosYawObs exposes StateSpace + MeasSpace typedefs"
```

---

## Task 6: Constrain UKF on concepts + add `Filter<UKF>` static_assert

**Files:**

- Modify: `cpp/include/immtrack/ukf.hpp`
- Modify: `tests/cpp/test_ukf_predict.cc`
- Modify: `tests/cpp/test_ukf_update.cc`
- Modify: `tests/cpp/test_ukf_predict_measurement.cc`

### Step 6.1: Update UKF tests for 8D state vectors

The existing UKF tests construct 7D state vectors. They must now construct 8D vectors (append `yaw_rate=0`).

- [ ] In `tests/cpp/test_ukf_predict.cc`, find all `Filter::StateVec` and `Filter::StateMat` literals and adjust:
  - Any `x << a, b, c, d, e, f, g;` (7 values) becomes `x << a, b, c, d, e, f, g, 0.0;` (8 values).
  - Any explicit dimension `7` becomes `8`.
  - The `theta = ukf.state()(6)` index stays the same (yaw is still at idx 6).

- [ ] Same updates in `tests/cpp/test_ukf_update.cc` and `tests/cpp/test_ukf_predict_measurement.cc`.

- [ ] Add an explicit concept check at the top of `tests/cpp/test_ukf_predict.cc`:

```cpp
#include <immtrack/concepts.hpp>
// ...
TEST_CASE("UKF<PosVxyzYawCV, PosYawObs> satisfies Filter concept", "[ukf]") {
  using F = immtrack::UKF<immtrack::PosVxyzYawCV, immtrack::PosYawObs>;
  STATIC_REQUIRE(immtrack::Filter<F>);
}
```

### Step 6.2: Run tests to see what fails

- [ ] Run:

```bash
cmake --build build-test -j 2>&1 | tail -30
```

Expected: tests compile, but the new `Filter<UKF<...>>` static_assert may fail because UKF lacks the `StateSpace`/`MeasSpace` typedefs the concept requires.

### Step 6.3: Add typedefs and concept constraints to `ukf.hpp`

- [ ] In `cpp/include/immtrack/ukf.hpp`, add the include and concept constraint:

Replace:

```cpp
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <immtrack/detail/sigma_points.hpp>
#include <immtrack/errors.hpp>
```

with:

```cpp
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <immtrack/concepts.hpp>
#include <immtrack/detail/sigma_points.hpp>
#include <immtrack/errors.hpp>
```

Replace:

```cpp
template <class Motion, class Obs>
class UKF {
 public:
  static constexpr int N = Motion::N;
```

with:

```cpp
template <MotionModel Motion, ObservationModel Obs>
  requires std::same_as<typename Motion::StateSpace, typename Obs::StateSpace>
class UKF {
 public:
  using StateSpace = typename Motion::StateSpace;
  using MeasSpace  = typename Obs::MeasSpace;
  static constexpr int N = StateSpace::state_dim;
```

Inside the same class, replace:

```cpp
  static constexpr int M = Obs::M;
```

with:

```cpp
  static constexpr int M = MeasSpace::state_dim;
```

- [ ] At the end of `ukf.hpp`, just before the closing `}  // namespace immtrack`, add:

```cpp
// Compile-time proof that UKF<PosVxyzYawCV, PosYawObs> satisfies the Filter concept.
// Lives behind an inclusion guard via forward typedefs so concrete motions/obs
// can opt in by including their headers and re-declaring as needed.
}  // namespace immtrack

#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
namespace immtrack {
static_assert(Filter<UKF<PosVxyzYawCV, PosYawObs>>,
              "UKF<PosVxyzYawCV, PosYawObs> must satisfy Filter concept");
}  // namespace immtrack
```

(Yes the close-then-reopen of the namespace looks ugly; it isolates the
heavy includes from the bare UKF definition. Acceptable for a
single-translation-unit static_assert.)

### Step 6.4: Run tests

- [ ] Run:

```bash
cmake --build build-test -j && ctest --test-dir build-test --output-on-failure
```

Expected: every test passes. Numerical results in UKF tests should match prior 7D values for indices 0..6; new index 7 values are 0 by initialisation.

### Step 6.5: Commit

- [ ] Run:

```bash
git add cpp/include/immtrack/ukf.hpp tests/cpp/test_ukf_predict.cc tests/cpp/test_ukf_update.cc tests/cpp/test_ukf_predict_measurement.cc
git commit -m "refactor(ukf): constrain on MotionModel/ObservationModel; add Filter<UKF> static_assert"
```

---

## Task 7: Update `Track::Track(...)` for 8D state init

**Files:**

- Modify: `cpp/include/immtrack/tracker.hpp`
- Modify: `tests/cpp/test_bbox_tracker.cc`

### Step 7.1: Inspect existing Track init

The existing `Track::Track(int id, const BoundingBox& d)` does:

```cpp
typename Filter::StateVec s;
s.setZero();           // already covers index 7 (yaw_rate=0)
s(0) = d.x;
s(1) = d.y;
s(2) = d.z;
s(6) = d.rot;
filter_.init(s, Filter::StateMat::Identity());
```

With 8D, `s.setZero()` and `Filter::StateMat::Identity()` automatically scale to the new dimension. **No code change is mechanically required**, but the default `P0 = I` puts variance 1.0 on the yaw_rate row/col, which is a reasonable Phase 1 default.

### Step 7.2: Add a test asserting yaw_rate column behavior

- [ ] Add to `tests/cpp/test_bbox_tracker.cc` (preserve existing tests):

```cpp
TEST_CASE("Track init places yaw_rate at idx 7 with finite variance", "[bbox_tracker]") {
  using namespace immtrack;
  using F = UKF<PosVxyzYawCV, PosYawObs>;
  BoundingBox d{};
  d.x = 1.0; d.y = 2.0; d.z = 3.0; d.rot = 0.4;
  d.l = 4.0; d.w = 1.8; d.h = 1.6; d.score = 0.9; d.class_name = "car";
  Track<F> t(/*id=*/0, d);
  const auto& x = t.filter().state();
  REQUIRE(x(F::StateSpace::YAW_RATE) == Catch::Approx(0.0));
  const auto& P = t.filter().covariance();
  REQUIRE(P(F::StateSpace::YAW_RATE, F::StateSpace::YAW_RATE) > 0.0);
}
```

### Step 7.3: Run tests

- [ ] Run:

```bash
cmake --build build-test -j --target test_bbox_tracker && ./build-test/tests/cpp/test_bbox_tracker
```

Expected: pass. If `Track<F>::filter()` is private, expose it via the existing public accessor (already present per `tracker.hpp` reading).

### Step 7.4: Commit

- [ ] Run:

```bash
git add tests/cpp/test_bbox_tracker.cc
git commit -m "test(tracker): assert Track init places yaw_rate at idx 7"
```

---

## Task 8: Verify Python bindings still link and pass

**Files:**

- Modify (if needed): `bindings/_core.cc`
- Verify: `tests/python/`

### Step 8.1: Build the pybind11 module

- [ ] Run:

```bash
cmake --build build-test -j --target _core 2>&1 | tail -20
```

Expected: clean build. No source change needed because `_core.cc` is templated on `Filter` and uses `Filter::N`, which now reports 8.

If any build error mentions a hardcoded `7`, fix it in `_core.cc` and rebuild.

### Step 8.2: Run Python tests

- [ ] Run:

```bash
pip install -e . --no-build-isolation
pytest tests/python/ -v 2>&1 | tail -40
```

Expected: tests pass. Tests that hardcode the state vector size to 7 must be updated to 8 and the position of `yaw_rate=0` appended where state vectors are constructed in fixtures.

### Step 8.3: Commit any python-side updates

- [ ] If anything was edited:

```bash
git add tests/python/ bindings/_core.cc
git commit -m "test(python): adjust fixtures to 8D state vector"
```

If nothing changed, skip the commit (the prior C++ commits already cover the change).

---

## Task 9: Full-suite acceptance + AMOTA equivalence smoke

**Files:**

- None modified. This is a verification gate.

### Step 9.1: Rebuild clean

- [ ] Run:

```bash
rm -rf build-test
cmake -S . -B build-test -DIMMTRACK_BUILD_TESTS=ON
cmake --build build-test -j
```

### Step 9.2: Run all C++ tests

- [ ] Run:

```bash
ctest --test-dir build-test --output-on-failure
```

Expected: all green. If any test fails, fix and re-run.

### Step 9.3: Run all Python tests

- [ ] Run:

```bash
pytest tests/python/ -v
```

Expected: all green.

### Step 9.4: Smoke-test AMOTA on a synthetic scenario

- [ ] Run:

```bash
pytest tests/python/test_amota.py -v
```

Expected: AMOTA score returned by the refactored tracker is within
**1e-6 absolute** of the score from `main` for the same input
sequence. Acceptance criterion: numerical equivalence. If divergence
exceeds threshold, investigate: most likely root cause is
`process_noise` differences (the new closed-form Q vs the old
`Cov::Identity() * dt`).

If the new `process_noise` formulation breaks parity, restore the
old `dt * I_{6x6}` block plus zero on yaw_rate row/col as a
parity-preserving fallback. Phase 2 will retune Q anyway when CTRV
lands.

### Step 9.5: Tag the Phase 1 milestone

- [ ] Run:

```bash
git tag -a phase1-concept-layer -m "Phase 1: concept layer + 8D state, IMM-ready"
```

### Step 9.6: Done

Phase 1 complete. Phase 2 (IMM with CTRV) builds on this foundation
without touching `Track`, `BBoxTracker`, or `bindings/_core.cc`.

---

## Self-Review Checklist (already addressed)

1. **Spec coverage:**
   - Add `concepts.hpp` + `detail/euclidean.hpp` → Tasks 1, 2.
   - Add `state_spaces.hpp` with `XYZVxVyVzYawYawRateSpace` → Task 3.
   - Refactor `PosVxyzYawCV` → Task 4.
   - Refactor `PosYawObs` → Task 5.
   - Refactor `UKF` (concept-constrained, `Filter<UKF>` static_assert) → Task 6.
   - `Track` init for 8D → Task 7.
   - Bindings + Python tests propagate → Task 8.
   - AMOTA equivalence → Task 9.
2. **Placeholder scan:** No "TBD"; all code blocks are concrete. The
   only narrative sections are the Self-Review and the comment in Task
   7 explaining why no Track code change is required.
3. **Type consistency:** `XYZVxVyVzYawYawRateSpace::YAW = 6` and
   `YAW_RATE = 7` referenced consistently across Tasks 3, 4, 5, 7.
   `MotionModel`/`ObservationModel`/`LinearMotion`/`LinearObs`/`Filter`
   spellings match across tasks. `StateSpace` and `MeasSpace` member
   names match the concept definitions in Task 1.
