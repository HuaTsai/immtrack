# immtrack Initial UKF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v0.1 placeholder UKF scaffold with a full unscented transform implementation specialized for `PosVxyzYawCV` motion (7D state `[x, y, z, vx, vy, vz, θ]`) and `PosYawObs` measurement (4D `[x, y, z, θ]`), exposed as `UkfPosVxyzYawCV` in Python.

**Architecture:** Trait-based static dispatch via `template <class Motion, class Obs> class UKF`. Compile-time matrix dimensions throughout (Eigen fixed-size). Angle handling via trait hooks (`weighted_mean`, `residual`). No virtual functions, no dynamic allocation in hot path.

**Tech Stack:** C++20, Eigen 3.4 (header-only), pybind11, scikit-build-core, Catch2 v3 (FetchContent), pytest.

**Spec reference:** `docs/superpowers/specs/2026-05-04-immtrack-initial-ukf-design.md`

---

## File Structure

**New files:**

- `cpp/include/immtrack/errors.hpp` — exception types
- `cpp/include/immtrack/detail/angle.hpp` — `wrap_angle`
- `cpp/include/immtrack/detail/sigma_points.hpp` — Merwe scaled generator + weights
- `tests/cpp/CMakeLists.txt` — Catch2 FetchContent + per-file test exe
- `tests/cpp/helpers/eigen_matchers.hpp` — `IsApprox`, `IsPsd`
- `tests/cpp/test_angle.cpp`
- `tests/cpp/test_sigma_points.cpp`
- `tests/cpp/test_traits.cpp`
- `tests/cpp/test_ukf_predict.cpp`
- `tests/cpp/test_ukf_update.cpp`

**Replaced (full rewrite):**

- `cpp/include/immtrack/motion.hpp` — only `PosVxyzYawCV` (delete `CV3D`, `CTRV3D`)
- `cpp/include/immtrack/observations.hpp` — only `PosYawObs` (delete `BBox3DObs`)
- `cpp/include/immtrack/ukf.hpp` — full unscented transform
- `bindings/_core.cc` — register `UkfPosVxyzYawCV` + 3 exceptions
- `src/immtrack/__init__.py` — re-export new symbols
- `tests/python/test_smoke.py` — match new dimensions and class

**Modified:**

- `CMakeLists.txt` — add `immtrack_core` INTERFACE library + `IMMTRACK_BUILD_TESTS` option

---

## Task 1: CMake foundation + Catch2 wiring

Set up the build system to support C++ tests without breaking the existing Python build. This task makes no functional change to runtime code.

**Files:**

- Modify: `CMakeLists.txt`
- Create: `tests/cpp/CMakeLists.txt`

- [ ] **Step 1: Replace top-level `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.15)
project(immtrack LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.4 CONFIG REQUIRED)

# Header-only core library — used by both the pybind module and tests.
add_library(immtrack_core INTERFACE)
target_include_directories(immtrack_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)
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

- [ ] **Step 2: Create `tests/cpp/CMakeLists.txt`**

```cmake
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

function(immtrack_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE
        immtrack_core
        Catch2::Catch2WithMain
    )
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers
    )
    catch_discover_tests(${name})
endfunction()

# Tests are added in subsequent tasks. The list grows as features land.
```

- [ ] **Step 3: Verify Python build still works (regression check)**

Run:

```bash
rm -rf build/ && uv sync
uv run python -c "from immtrack import CtrvBBox3DUKF, CvBBox3DUKF; print('OK')"
uv run pytest tests/python -q
```

Expected: build succeeds, existing imports still work, all 5 existing pytest cases pass. (We have not changed any runtime code yet.)

- [ ] **Step 4: Verify test build infrastructure configures**

Run:

```bash
cmake -S . -B build-test -DIMMTRACK_BUILD_TESTS=ON
```

Expected: configure succeeds. No tests yet, but Catch2 should download. Check `build-test/_deps/catch2-src/` exists.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt tests/cpp/CMakeLists.txt
git commit -m "build: add immtrack_core INTERFACE lib and Catch2 test harness

- Extract header-only INTERFACE target from pybind module so tests can
  link the same headers directly.
- Add IMMTRACK_BUILD_TESTS option (default OFF) so pip install/uv build
  do not download Catch2.
- Wire tests/cpp via FetchContent for Catch2 v3.5.4."
```

---

## Task 2: `wrap_angle` helper (TDD)

Foundation utility used by trait residual implementations. Pure function, no dependencies.

**Files:**

- Create: `cpp/include/immtrack/detail/angle.hpp`
- Create: `tests/cpp/test_angle.cpp`
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create `tests/cpp/test_angle.cpp`:

```cpp
#include <cmath>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <immtrack/detail/angle.hpp>

using Catch::Matchers::WithinAbs;
using immtrack::detail::wrap_angle;

TEST_CASE("wrap_angle: in-range values unchanged", "[angle]") {
    REQUIRE_THAT(wrap_angle(0.0), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(wrap_angle(1.0), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(wrap_angle(-1.0), WithinAbs(-1.0, 1e-12));
}

TEST_CASE("wrap_angle: just past +pi wraps to negative", "[angle]") {
    const double a = M_PI + 0.1;
    REQUIRE_THAT(wrap_angle(a), WithinAbs(-M_PI + 0.1, 1e-12));
}

TEST_CASE("wrap_angle: just past -pi wraps to positive", "[angle]") {
    const double a = -M_PI - 0.1;
    REQUIRE_THAT(wrap_angle(a), WithinAbs(M_PI - 0.1, 1e-12));
}

TEST_CASE("wrap_angle: multi-revolution input", "[angle]") {
    REQUIRE_THAT(wrap_angle(4 * M_PI + 0.5), WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(wrap_angle(-4 * M_PI - 0.5), WithinAbs(-0.5, 1e-12));
}

TEST_CASE("wrap_angle: NaN propagates to NaN", "[angle]") {
    REQUIRE(std::isnan(wrap_angle(std::numeric_limits<double>::quiet_NaN())));
}
```

Add `immtrack_add_test(test_angle)` to `tests/cpp/CMakeLists.txt`.

- [ ] **Step 2: Run test to verify it fails (header missing)**

Run:

```bash
cmake -S . -B build-test -DIMMTRACK_BUILD_TESTS=ON
cmake --build build-test --target test_angle
```

Expected: compile error — `immtrack/detail/angle.hpp: No such file or directory`.

- [ ] **Step 3: Implement `wrap_angle`**

Create `cpp/include/immtrack/detail/angle.hpp`:

```cpp
#pragma once

#include <cmath>

namespace immtrack::detail {

// Wrap angle to [-pi, pi]. Uses atan2(sin, cos) for robustness against
// large inputs and IEEE 754 special values; fmod is implementation-defined
// for negative operands and prone to round-off near boundaries.
inline double wrap_angle(double a) noexcept {
    return std::atan2(std::sin(a), std::cos(a));
}

}  // namespace immtrack::detail
```

- [ ] **Step 4: Run test, verify pass**

Run:

```bash
cmake --build build-test --target test_angle
ctest --test-dir build-test -R test_angle --output-on-failure
```

Expected: 5 test cases pass.

- [ ] **Step 5: Commit**

```bash
git add cpp/include/immtrack/detail/angle.hpp \
        tests/cpp/test_angle.cpp \
        tests/cpp/CMakeLists.txt
git commit -m "feat: add wrap_angle helper

Wraps an angle into [-pi, pi] via atan2(sin, cos) to avoid
implementation-defined fmod behaviour and round-off at boundaries."
```

---

## Task 3: Eigen test matchers

Custom Catch2 matchers used by sigma point and UKF tests. Adding them up front so later tasks can use them without scope creep.

**Files:**

- Create: `tests/cpp/helpers/eigen_matchers.hpp`
- Create: `tests/cpp/test_matchers_smoke.cpp` (deleted at end of this task)
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 1: Write a smoke test that uses both matchers**

Create `tests/cpp/test_matchers_smoke.cpp`:

```cpp
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>

#include "eigen_matchers.hpp"

using immtrack::test::IsApprox;
using immtrack::test::IsPsd;

TEST_CASE("IsApprox passes on equal matrices", "[matchers]") {
    Eigen::Matrix3d a = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d b = Eigen::Matrix3d::Identity();
    REQUIRE_THAT(a, IsApprox(b, 1e-12));
}

TEST_CASE("IsApprox fails on different matrices", "[matchers]") {
    Eigen::Matrix3d a = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d b = Eigen::Matrix3d::Identity() * 2.0;
    REQUIRE_FALSE(IsApprox(b, 1e-12).match(a));
}

TEST_CASE("IsPsd passes on identity", "[matchers]") {
    Eigen::Matrix3d a = Eigen::Matrix3d::Identity();
    REQUIRE_THAT(a, IsPsd());
}

TEST_CASE("IsPsd fails on negative-definite", "[matchers]") {
    Eigen::Matrix3d a = -Eigen::Matrix3d::Identity();
    REQUIRE_FALSE(IsPsd().match(a));
}
```

Add `immtrack_add_test(test_matchers_smoke)` to `tests/cpp/CMakeLists.txt`.

- [ ] **Step 2: Run, verify failure (header missing)**

```bash
cmake --build build-test --target test_matchers_smoke
```

Expected: compile error — `eigen_matchers.hpp` not found.

- [ ] **Step 3: Implement matchers**

Create `tests/cpp/helpers/eigen_matchers.hpp`:

```cpp
#pragma once

#include <sstream>
#include <string>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <catch2/matchers/catch_matchers_templated.hpp>

namespace immtrack::test {

// Approximate-equality matcher for any Eigen dense matrix.
class IsApprox : public Catch::Matchers::MatcherGenericBase {
   public:
    IsApprox(Eigen::MatrixXd expected, double tol)
        : expected_(std::move(expected)), tol_(tol) {}

    template <class Derived>
    bool match(const Eigen::MatrixBase<Derived>& actual) const {
        if (actual.rows() != expected_.rows() ||
            actual.cols() != expected_.cols()) {
            return false;
        }
        return (actual - expected_).norm() < tol_;
    }

    std::string describe() const override {
        std::ostringstream oss;
        oss << "is approximately equal to expected matrix (tol = " << tol_
            << ")";
        return oss.str();
    }

   private:
    Eigen::MatrixXd expected_;
    double tol_;
};

// Positive-semi-definite matcher: all eigenvalues >= -tol.
class IsPsd : public Catch::Matchers::MatcherGenericBase {
   public:
    explicit IsPsd(double tol = 1e-10) : tol_(tol) {}

    template <class Derived>
    bool match(const Eigen::MatrixBase<Derived>& actual) const {
        if (actual.rows() != actual.cols()) {
            return false;
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(actual);
        if (solver.info() != Eigen::Success) {
            return false;
        }
        return solver.eigenvalues().minCoeff() >= -tol_;
    }

    std::string describe() const override {
        std::ostringstream oss;
        oss << "is positive semi-definite (tol = " << tol_ << ")";
        return oss.str();
    }

   private:
    double tol_;
};

}  // namespace immtrack::test
```

- [ ] **Step 4: Run smoke test, verify pass**

```bash
cmake --build build-test --target test_matchers_smoke
ctest --test-dir build-test -R test_matchers_smoke --output-on-failure
```

Expected: 4 test cases pass.

- [ ] **Step 5: Delete the smoke test** (it was just to validate the matchers; later tests will exercise them implicitly)

```bash
rm tests/cpp/test_matchers_smoke.cpp
```

Remove the `immtrack_add_test(test_matchers_smoke)` line from `tests/cpp/CMakeLists.txt`.

- [ ] **Step 6: Commit**

```bash
git add tests/cpp/helpers/eigen_matchers.hpp tests/cpp/CMakeLists.txt
git commit -m "test: add IsApprox and IsPsd Eigen matchers for Catch2

Used by sigma-point and UKF tests in subsequent tasks."
```

---

## Task 4: Sigma points helper (TDD)

Merwe scaled sigma point generator with LLT/LDLT fallback, plus the weights struct. Tests verify that sigma points reconstruct the input mean and covariance.

**Files:**

- Create: `cpp/include/immtrack/errors.hpp` (used by sigma points on PSD failure)
- Create: `cpp/include/immtrack/detail/sigma_points.hpp`
- Create: `tests/cpp/test_sigma_points.cpp`
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 1: Create `errors.hpp`**

Create `cpp/include/immtrack/errors.hpp`:

```cpp
#pragma once

#include <stdexcept>

namespace immtrack {

class CovarianceNotPsd : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

class InvalidArgument : public std::invalid_argument {
   public:
    using std::invalid_argument::invalid_argument;
};

class NumericalError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

}  // namespace immtrack
```

- [ ] **Step 2: Write the failing tests**

Create `tests/cpp/test_sigma_points.cpp`:

```cpp
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <immtrack/detail/sigma_points.hpp>
#include <immtrack/errors.hpp>

#include "eigen_matchers.hpp"

using immtrack::test::IsApprox;
using immtrack::detail::generate_sigma_points;
using immtrack::detail::UnscentedWeights;

TEMPLATE_TEST_CASE_SIG("Sigma points reconstruct mean and cov", "[sigma]",
                       ((int N), N), 3, 5, 7, 10) {
    using Vec = Eigen::Matrix<double, N, 1>;
    using Mat = Eigen::Matrix<double, N, N>;

    Vec mu = Vec::LinSpaced(1.0, static_cast<double>(N));
    Mat sigma = Mat::Identity() * 0.5;

    constexpr double alpha = 1e-3;
    constexpr double beta = 2.0;
    constexpr double kappa = 0.0;
    const auto weights = UnscentedWeights<N>::make(alpha, beta, kappa);

    const auto points = generate_sigma_points<N>(mu, sigma, weights.lambda);

    // Reconstruct mean: must equal mu (linear combination of symmetric points).
    Vec reconstructed_mean = Vec::Zero();
    for (int i = 0; i < 2 * N + 1; ++i) {
        reconstructed_mean += weights.mean_weights(i) * points.col(i);
    }
    REQUIRE_THAT(reconstructed_mean, IsApprox(mu, 1e-10));

    // Reconstruct covariance.
    Mat reconstructed_cov = Mat::Zero();
    for (int i = 0; i < 2 * N + 1; ++i) {
        const Vec d = points.col(i) - mu;
        reconstructed_cov += weights.cov_weights(i) * d * d.transpose();
    }
    REQUIRE_THAT(reconstructed_cov, IsApprox(sigma, 1e-10));
}

TEST_CASE("Sigma point 0 equals mean", "[sigma]") {
    constexpr int N = 4;
    using Vec = Eigen::Matrix<double, N, 1>;
    using Mat = Eigen::Matrix<double, N, N>;

    Vec mu;
    mu << 1.0, -2.0, 3.5, 0.0;
    Mat sigma = Mat::Identity();

    const auto weights = UnscentedWeights<N>::make(1e-3, 2.0, 0.0);
    const auto points = generate_sigma_points<N>(mu, sigma, weights.lambda);

    REQUIRE_THAT(points.col(0), IsApprox(mu, 1e-12));
}

TEST_CASE("Sigma points are symmetric around mean", "[sigma]") {
    constexpr int N = 3;
    using Vec = Eigen::Matrix<double, N, 1>;
    using Mat = Eigen::Matrix<double, N, N>;

    Vec mu = Vec::Zero();
    Mat sigma = Mat::Identity() * 2.0;
    const auto weights = UnscentedWeights<N>::make(1e-3, 2.0, 0.0);
    const auto points = generate_sigma_points<N>(mu, sigma, weights.lambda);

    for (int i = 0; i < N; ++i) {
        const Vec sum = points.col(i + 1) + points.col(i + 1 + N);
        REQUIRE_THAT(sum, IsApprox(2.0 * mu, 1e-12));
    }
}

TEST_CASE("generate_sigma_points throws on negative-definite cov", "[sigma]") {
    constexpr int N = 3;
    using Vec = Eigen::Matrix<double, N, 1>;
    using Mat = Eigen::Matrix<double, N, N>;

    Vec mu = Vec::Zero();
    Mat sigma = -Mat::Identity();
    const auto weights = UnscentedWeights<N>::make(1e-3, 2.0, 0.0);

    REQUIRE_THROWS_AS(
        generate_sigma_points<N>(mu, sigma, weights.lambda),
        immtrack::CovarianceNotPsd);
}

TEST_CASE("UnscentedWeights row sums equal 1 for mean weights", "[sigma]") {
    const auto weights = UnscentedWeights<7>::make(1e-3, 2.0, 0.0);
    REQUIRE_THAT(weights.mean_weights.sum(),
                 Catch::Matchers::WithinAbs(1.0, 1e-10));
}
```

Add `immtrack_add_test(test_sigma_points)` to `tests/cpp/CMakeLists.txt`.

- [ ] **Step 3: Run, verify failures (header missing)**

```bash
cmake --build build-test --target test_sigma_points 2>&1 | head -20
```

Expected: compile error — `immtrack/detail/sigma_points.hpp` not found.

- [ ] **Step 4: Implement sigma points helper**

Create `cpp/include/immtrack/detail/sigma_points.hpp`:

```cpp
#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <immtrack/errors.hpp>

namespace immtrack::detail {

// Merwe scaled unscented weights.
//   lambda = alpha^2 * (N + kappa) - N
//   W_m_0 = lambda / (N + lambda)
//   W_c_0 = lambda / (N + lambda) + (1 - alpha^2 + beta)
//   W_m_i = W_c_i = 1 / (2 (N + lambda))   for i = 1..2N
template <int N>
struct UnscentedWeights {
    static constexpr int K = 2 * N + 1;
    using Vec = Eigen::Matrix<double, K, 1>;

    Vec mean_weights;
    Vec cov_weights;
    double lambda;

    static UnscentedWeights make(double alpha, double beta, double kappa) {
        UnscentedWeights w{};
        w.lambda = alpha * alpha * (N + kappa) - N;
        const double denom = static_cast<double>(N) + w.lambda;
        const double rest = 1.0 / (2.0 * denom);
        w.mean_weights(0) = w.lambda / denom;
        w.cov_weights(0) = w.lambda / denom + (1.0 - alpha * alpha + beta);
        for (int i = 1; i < K; ++i) {
            w.mean_weights(i) = rest;
            w.cov_weights(i) = rest;
        }
        return w;
    }
};

// Compute lower-triangular square root of M via LLT, falling back to LDLT
// (with negative D clamped to zero). Throws CovarianceNotPsd if both fail.
template <int N>
Eigen::Matrix<double, N, N> safe_cholesky(
    const Eigen::Matrix<double, N, N>& M) {
    using Mat = Eigen::Matrix<double, N, N>;

    Eigen::LLT<Mat> llt(M);
    if (llt.info() == Eigen::Success) {
        return llt.matrixL();
    }

    Eigen::LDLT<Mat> ldlt(M);
    if (ldlt.info() != Eigen::Success) {
        throw CovarianceNotPsd(
            "safe_cholesky: matrix not positive semi-definite");
    }
    // Reject if D has any meaningfully negative entry (matrix is indefinite).
    if (ldlt.vectorD().minCoeff() < -1e-12) {
        throw CovarianceNotPsd(
            "safe_cholesky: matrix has negative eigenvalues");
    }
    // M = P^T L D L^T P  =>  S = P^T L sqrt(D) satisfies S S^T = M.
    // Without applying transpositionsP^T, S would reconstruct P M P^T, not M.
    const auto D = ldlt.vectorD().array().max(0.0).sqrt().matrix();
    Mat unpermuted = Mat(ldlt.matrixL()) * D.asDiagonal();
    return ldlt.transpositionsP().transpose() * unpermuted;
}

// Generate 2N+1 Merwe scaled sigma points from mean mu and covariance sigma.
// Columns are: chi_0 = mu, chi_i = mu + sqrt((N+lambda) sigma)_i,
//              chi_{N+i} = mu - sqrt((N+lambda) sigma)_i.
template <int N>
Eigen::Matrix<double, N, 2 * N + 1> generate_sigma_points(
    const Eigen::Matrix<double, N, 1>& mu,
    const Eigen::Matrix<double, N, N>& sigma,
    double lambda) {
    using Mat = Eigen::Matrix<double, N, N>;
    using Points = Eigen::Matrix<double, N, 2 * N + 1>;

    const Mat scaled = (static_cast<double>(N) + lambda) * sigma;
    const Mat L = safe_cholesky<N>(scaled);

    Points points;
    points.col(0) = mu;
    for (int i = 0; i < N; ++i) {
        points.col(i + 1) = mu + L.col(i);
        points.col(i + 1 + N) = mu - L.col(i);
    }
    return points;
}

}  // namespace immtrack::detail
```

- [ ] **Step 5: Run, verify pass**

```bash
cmake --build build-test --target test_sigma_points
ctest --test-dir build-test -R test_sigma_points --output-on-failure
```

Expected: all test cases pass (4 dimensions × 1 template case + 4 single cases = 8 cases).

- [ ] **Step 6: Commit**

```bash
git add cpp/include/immtrack/errors.hpp \
        cpp/include/immtrack/detail/sigma_points.hpp \
        tests/cpp/test_sigma_points.cpp \
        tests/cpp/CMakeLists.txt
git commit -m "feat: Merwe scaled sigma points + LLT/LDLT fallback

- errors.hpp introduces CovarianceNotPsd / InvalidArgument /
  NumericalError exception types.
- detail/sigma_points.hpp provides generate_sigma_points<N>() and
  UnscentedWeights<N>::make(alpha, beta, kappa).
- safe_cholesky falls back to LDLT with D clamped to zero before
  raising CovarianceNotPsd."
```

---

## Task 5: API transition — replace types and rewire bindings

This task atomically renames `CV3D` / `CTRV3D` / `BBox3DObs` / `CvBBox3DUKF` / `CtrvBBox3DUKF` to the new spec types in a single commit. The `UKF::predict` and `UKF::update` are still placeholders here; Task 7 and 8 implement them. After this commit, the Python module exposes `UkfPosVxyzYawCV` with shape-correct but numerically incomplete behaviour.

**Files:**

- Modify: `cpp/include/immtrack/motion.hpp` (full rewrite)
- Modify: `cpp/include/immtrack/observations.hpp` (full rewrite)
- Modify: `cpp/include/immtrack/ukf.hpp` (constructor signature only; predict/update still placeholders)
- Modify: `bindings/_core.cc` (full rewrite)
- Modify: `src/immtrack/__init__.py` (full rewrite)
- Modify: `tests/python/test_smoke.py` (full rewrite)
- Create: `tests/cpp/test_traits.cpp`
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 1: Replace `cpp/include/immtrack/motion.hpp`**

```cpp
#pragma once

#include <cmath>

#include <Eigen/Core>

#include <immtrack/detail/angle.hpp>

namespace immtrack {

// Constant-velocity model in 3D Cartesian space with a passive yaw
// dimension.
//   state = [x, y, z, vx, vy, vz, theta]   (N = 7)
// Yaw is carried as part of the state but does not evolve under CV
// dynamics; it is updated only via measurement.
struct PosVxyzYawCV {
    static constexpr int N = 7;
    using State = Eigen::Matrix<double, N, 1>;
    using Cov = Eigen::Matrix<double, N, N>;

    static State predict(const State& x, double dt) {
        State next = x;
        next(0) += x(3) * dt;  // x  += vx * dt
        next(1) += x(4) * dt;  // y  += vy * dt
        next(2) += x(5) * dt;  // z  += vz * dt
        // velocities and theta unchanged under CV
        return next;
    }

    static Cov process_noise(double dt) {
        // Placeholder: identity scaled by dt. Tune in a later spec.
        return Cov::Identity() * dt;
    }

    // Reconstruct mean over sigma points. Uses arithmetic average for
    // linear dimensions and circular mean (atan2 of weighted sin/cos)
    // for theta at index 6.
    template <int K>
    static State weighted_mean(
        const Eigen::Matrix<double, N, K>& sigmas,
        const Eigen::Matrix<double, K, 1>& weights) {
        State mean = State::Zero();
        for (int i = 0; i < K; ++i) {
            mean.template head<6>() +=
                weights(i) * sigmas.col(i).template head<6>();
        }
        double sin_sum = 0.0;
        double cos_sum = 0.0;
        for (int i = 0; i < K; ++i) {
            sin_sum += weights(i) * std::sin(sigmas(6, i));
            cos_sum += weights(i) * std::cos(sigmas(6, i));
        }
        mean(6) = std::atan2(sin_sum, cos_sum);
        return mean;
    }

    // Compute a - b with theta-component wrapped to [-pi, pi].
    static State residual(const State& a, const State& b) {
        State r = a - b;
        r(6) = detail::wrap_angle(r(6));
        return r;
    }
};

}  // namespace immtrack
```

- [ ] **Step 2: Replace `cpp/include/immtrack/observations.hpp`**

```cpp
#pragma once

#include <cmath>

#include <Eigen/Core>

#include <immtrack/detail/angle.hpp>

namespace immtrack {

// 3D position + yaw observation:
//   measurement = [x, y, z, theta]   (M = 4)
// Compatible with any motion whose state has x at idx 0..2 and theta at
// idx 6 (e.g. PosVxyzYawCV).
struct PosYawObs {
    static constexpr int M = 4;
    using Meas = Eigen::Matrix<double, M, 1>;
    using Noise = Eigen::Matrix<double, M, M>;

    template <class State>
    static Meas h(const State& x) {
        Meas z;
        z(0) = x(0);
        z(1) = x(1);
        z(2) = x(2);
        z(3) = x(6);
        return z;
    }

    static Noise measurement_noise() {
        // Placeholder: identity. Tune per sensor in a later spec.
        return Noise::Identity();
    }

    template <int K>
    static Meas weighted_mean(
        const Eigen::Matrix<double, M, K>& sigmas,
        const Eigen::Matrix<double, K, 1>& weights) {
        Meas mean = Meas::Zero();
        for (int i = 0; i < K; ++i) {
            mean.template head<3>() +=
                weights(i) * sigmas.col(i).template head<3>();
        }
        double sin_sum = 0.0;
        double cos_sum = 0.0;
        for (int i = 0; i < K; ++i) {
            sin_sum += weights(i) * std::sin(sigmas(3, i));
            cos_sum += weights(i) * std::cos(sigmas(3, i));
        }
        mean(3) = std::atan2(sin_sum, cos_sum);
        return mean;
    }

    static Meas residual(const Meas& a, const Meas& b) {
        Meas r = a - b;
        r(3) = detail::wrap_angle(r(3));
        return r;
    }
};

}  // namespace immtrack
```

- [ ] **Step 3: Update `cpp/include/immtrack/ukf.hpp` constructor signature**

The Task 7 and 8 will fully implement `predict` and `update`. For now, the constructor and accessor surface need to match the new spec; predict propagates the mean only (placeholder) and update throws.

```cpp
#pragma once

#include <stdexcept>

#include <Eigen/Core>

#include <immtrack/detail/sigma_points.hpp>
#include <immtrack/errors.hpp>

namespace immtrack {

// Unscented Kalman Filter parameterised by motion model + observation
// traits. The Motion / Obs traits document the trait contract that any
// new specialisation must satisfy.
template <class Motion, class Obs>
class UKF {
   public:
    static constexpr int N = Motion::N;
    static constexpr int M = Obs::M;
    static constexpr int K = 2 * N + 1;
    using StateVec = Eigen::Matrix<double, N, 1>;
    using StateMat = Eigen::Matrix<double, N, N>;
    using MeasVec = Eigen::Matrix<double, M, 1>;

    UKF() : UKF(1e-3, 2.0, 0.0) {}

    UKF(double alpha, double beta, double kappa)
        : x_(StateVec::Zero()),
          P_(StateMat::Identity()),
          weights_(detail::UnscentedWeights<N>::make(alpha, beta, kappa)) {}

    void init(const StateVec& x, const StateMat& P) {
        x_ = x;
        P_ = P;
    }

    // Placeholder propagation: mean-only advance via Motion::predict and
    // additive process noise. Full sigma-point predict lands in Task 7.
    void predict(double dt) {
        if (dt < 0.0) {
            throw InvalidArgument("UKF::predict: dt must be non-negative");
        }
        x_ = Motion::predict(x_, dt);
        P_ += Motion::process_noise(dt);
    }

    // Placeholder. Full unscented update lands in Task 8.
    double update(const MeasVec& /*z*/) {
        throw std::runtime_error("UKF::update not implemented yet");
    }

    const StateVec& state() const noexcept { return x_; }
    const StateMat& covariance() const noexcept { return P_; }

   private:
    StateVec x_;
    StateMat P_;
    detail::UnscentedWeights<N> weights_;
};

}  // namespace immtrack
```

- [ ] **Step 4: Replace `bindings/_core.cc`**

```cpp
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <immtrack/errors.hpp>
#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <immtrack/ukf.hpp>

namespace py = pybind11;

namespace {

template <class Filter>
void bind_filter(py::module_& m, const char* name) {
    py::class_<Filter>(m, name)
        .def(py::init<>())
        .def(py::init<double, double, double>(),
             py::arg("alpha") = 1e-3,
             py::arg("beta") = 2.0,
             py::arg("kappa") = 0.0)
        .def("init", &Filter::init,
             py::arg("state"), py::arg("cov"))
        .def("predict", &Filter::predict,
             py::arg("dt"))
        .def("update", &Filter::update,
             py::arg("measurement"),
             "Returns NIS (normalized innovation squared, scalar).")
        .def_property_readonly(
            "state",
            [](const Filter& f) { return typename Filter::StateVec(f.state()); })
        .def_property_readonly(
            "covariance",
            [](const Filter& f) {
                return typename Filter::StateMat(f.covariance());
            })
        .def_property_readonly_static(
            "N", [](py::object) { return Filter::N; })
        .def_property_readonly_static(
            "M", [](py::object) { return Filter::M; });
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "immtrack core bindings";

    py::register_exception<immtrack::CovarianceNotPsd>(m, "CovarianceNotPsd");
    py::register_exception<immtrack::InvalidArgument>(
        m, "InvalidArgument", PyExc_ValueError);
    py::register_exception<immtrack::NumericalError>(m, "NumericalError");

    using UkfPosVxyzYawCV =
        immtrack::UKF<immtrack::PosVxyzYawCV, immtrack::PosYawObs>;
    bind_filter<UkfPosVxyzYawCV>(m, "UkfPosVxyzYawCV");
}
```

- [ ] **Step 5: Replace `src/immtrack/__init__.py`**

```python
from immtrack._core import (
    CovarianceNotPsd,
    InvalidArgument,
    NumericalError,
    UkfPosVxyzYawCV,
)

__all__ = [
    "CovarianceNotPsd",
    "InvalidArgument",
    "NumericalError",
    "UkfPosVxyzYawCV",
]
```

- [ ] **Step 6: Replace `tests/python/test_smoke.py`**

```python
import numpy as np
import pytest

from immtrack import (
    CovarianceNotPsd,
    InvalidArgument,
    NumericalError,
    UkfPosVxyzYawCV,
)


def test_default_construction() -> None:
    f = UkfPosVxyzYawCV()
    assert f.state.shape == (7,)
    assert f.covariance.shape == (7, 7)
    np.testing.assert_array_equal(f.state, np.zeros(7))
    np.testing.assert_array_equal(f.covariance, np.eye(7))


def test_class_level_dimensions() -> None:
    assert UkfPosVxyzYawCV.N == 7
    assert UkfPosVxyzYawCV.M == 4


def test_init_round_trip() -> None:
    f = UkfPosVxyzYawCV()
    x0 = np.arange(7, dtype=np.float64)
    p0 = np.eye(7) * 2.0
    f.init(state=x0, cov=p0)
    np.testing.assert_array_equal(f.state, x0)
    np.testing.assert_array_equal(f.covariance, p0)


def test_predict_advances_position_under_cv() -> None:
    # state = [x, y, z, vx, vy, vz, theta], CV dynamics
    f = UkfPosVxyzYawCV()
    x0 = np.zeros(7)
    x0[3] = 1.0  # vx = 1 m/s
    f.init(state=x0, cov=np.eye(7))
    f.predict(dt=0.5)
    assert f.state[0] == pytest.approx(0.5)
    assert f.state[1] == 0.0
    assert f.state[2] == 0.0


def test_predict_rejects_negative_dt() -> None:
    f = UkfPosVxyzYawCV()
    f.init(state=np.zeros(7), cov=np.eye(7))
    with pytest.raises(ValueError):
        f.predict(dt=-0.1)


def test_update_not_yet_implemented_in_this_commit() -> None:
    # Update lands in Task 8. Until then it raises.
    f = UkfPosVxyzYawCV()
    f.init(state=np.zeros(7), cov=np.eye(7))
    with pytest.raises(RuntimeError):
        f.update(measurement=np.zeros(4))


def test_exception_classes_importable() -> None:
    assert issubclass(InvalidArgument, ValueError)
    assert issubclass(CovarianceNotPsd, RuntimeError)
    assert issubclass(NumericalError, RuntimeError)
```

- [ ] **Step 7: Add `tests/cpp/test_traits.cpp`**

Verify trait shapes at compile time so any future regression that changes a constexpr is caught immediately.

```cpp
#include <type_traits>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>

#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>

using immtrack::PosVxyzYawCV;
using immtrack::PosYawObs;

TEST_CASE("PosVxyzYawCV trait shape", "[traits]") {
    STATIC_REQUIRE(PosVxyzYawCV::N == 7);
    STATIC_REQUIRE(
        std::is_same_v<PosVxyzYawCV::State, Eigen::Matrix<double, 7, 1>>);
    STATIC_REQUIRE(
        std::is_same_v<PosVxyzYawCV::Cov, Eigen::Matrix<double, 7, 7>>);
}

TEST_CASE("PosYawObs trait shape", "[traits]") {
    STATIC_REQUIRE(PosYawObs::M == 4);
    STATIC_REQUIRE(
        std::is_same_v<PosYawObs::Meas, Eigen::Matrix<double, 4, 1>>);
    STATIC_REQUIRE(
        std::is_same_v<PosYawObs::Noise, Eigen::Matrix<double, 4, 4>>);
}

TEST_CASE("PosVxyzYawCV::predict propagates position", "[traits]") {
    PosVxyzYawCV::State x = PosVxyzYawCV::State::Zero();
    x(3) = 2.0;  // vx
    const auto next = PosVxyzYawCV::predict(x, 0.5);
    REQUIRE(next(0) == 1.0);
    REQUIRE(next(1) == 0.0);
    REQUIRE(next(3) == 2.0);  // velocity unchanged
}

TEST_CASE("PosYawObs::h projects position+yaw", "[traits]") {
    PosVxyzYawCV::State x;
    x << 1.0, 2.0, 3.0, 0.5, 0.0, 0.0, 1.57;
    const auto z = PosYawObs::h(x);
    REQUIRE(z(0) == 1.0);
    REQUIRE(z(1) == 2.0);
    REQUIRE(z(2) == 3.0);
    REQUIRE(z(3) == 1.57);
}
```

Add `immtrack_add_test(test_traits)` to `tests/cpp/CMakeLists.txt`.

- [ ] **Step 8: Build and run all tests**

```bash
# Python build first
rm -rf build/ && uv sync
uv run pytest tests/python -q

# C++ tests
cmake --build build-test --target test_traits test_angle test_sigma_points
ctest --test-dir build-test --output-on-failure
```

Expected:

- Python: 7 tests pass.
- C++: 3 test executables run, all cases pass.

- [ ] **Step 9: Commit**

```bash
git add cpp/include/immtrack/motion.hpp \
        cpp/include/immtrack/observations.hpp \
        cpp/include/immtrack/ukf.hpp \
        bindings/_core.cc \
        src/immtrack/__init__.py \
        tests/python/test_smoke.py \
        tests/cpp/test_traits.cpp \
        tests/cpp/CMakeLists.txt
git commit -m "feat!: replace v0.1 scaffold with PosVxyzYawCV + PosYawObs API

Rename + reshape:
- CV3D / CTRV3D (10D, with size) -> PosVxyzYawCV (7D, no size).
- BBox3DObs (7D) -> PosYawObs (4D, no size).
- UKF gains alpha/beta/kappa constructor with sane defaults.
- Python module exposes UkfPosVxyzYawCV plus 3 exception types.

UKF::predict still mean-only and UKF::update still throws; full
unscented transform lands in subsequent commits."
```

---

## Task 6: UKF::predict full unscented transform (TDD)

Replace the placeholder `UKF::predict` with the real sigma-point propagation.

**Files:**

- Modify: `cpp/include/immtrack/ukf.hpp`
- Create: `tests/cpp/test_ukf_predict.cpp`
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing tests**

Create `tests/cpp/test_ukf_predict.cpp`:

```cpp
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <immtrack/errors.hpp>
#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <immtrack/ukf.hpp>

#include "eigen_matchers.hpp"

using Catch::Matchers::WithinAbs;
using immtrack::PosVxyzYawCV;
using immtrack::PosYawObs;
using immtrack::UKF;
using immtrack::test::IsApprox;
using immtrack::test::IsPsd;

using Filter = UKF<PosVxyzYawCV, PosYawObs>;

TEST_CASE("Predict on linear motion matches plain motion model", "[predict]") {
    // For linear dynamics with linear weighted_mean (non-angular dims),
    // UKF predict mean must equal Motion::predict(mu, dt).
    Filter ukf;
    Filter::StateVec x0;
    x0 << 0.0, 0.0, 0.0, 1.0, 2.0, 0.5, 0.0;  // velocity in xyz, theta=0
    Filter::StateMat P0 = Filter::StateMat::Identity() * 0.1;
    ukf.init(x0, P0);

    ukf.predict(0.5);

    Filter::StateVec expected;
    expected << 0.5, 1.0, 0.25, 1.0, 2.0, 0.5, 0.0;
    REQUIRE_THAT(ukf.state(), IsApprox(expected, 1e-9));
}

TEST_CASE("Predict with dt = 0 leaves state approximately unchanged",
          "[predict]") {
    Filter ukf;
    Filter::StateVec x0;
    x0 << 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4;
    Filter::StateMat P0 = Filter::StateMat::Identity() * 0.5;
    ukf.init(x0, P0);

    ukf.predict(0.0);

    REQUIRE_THAT(ukf.state(), IsApprox(x0, 1e-9));
}

TEST_CASE("Predict with negative dt throws InvalidArgument", "[predict]") {
    Filter ukf;
    ukf.init(Filter::StateVec::Zero(), Filter::StateMat::Identity());
    REQUIRE_THROWS_AS(ukf.predict(-0.1), immtrack::InvalidArgument);
}

TEST_CASE("Predict preserves PSD over many cycles", "[predict]") {
    Filter ukf;
    Filter::StateVec x0 = Filter::StateVec::Zero();
    x0(3) = 1.0;
    Filter::StateMat P0 = Filter::StateMat::Identity() * 0.1;
    ukf.init(x0, P0);

    for (int i = 0; i < 1000; ++i) {
        ukf.predict(0.05);
    }

    REQUIRE_THAT(ukf.covariance(), IsPsd());
}

TEST_CASE("Predict handles theta crossing +pi via circular mean",
          "[predict]") {
    // Theta near +pi; sigma points may straddle the boundary. Circular
    // mean must keep mean within (-pi, pi] without 2*pi jump.
    Filter ukf;
    Filter::StateVec x0 = Filter::StateVec::Zero();
    x0(6) = M_PI - 0.01;  // theta just below +pi
    Filter::StateMat P0 = Filter::StateMat::Identity() * 0.5;
    P0(6, 6) = 0.5;  // wide theta uncertainty; sigma points will straddle pi
    ukf.init(x0, P0);

    ukf.predict(0.01);

    // Theta should remain in [-pi, pi].
    const double theta = ukf.state()(6);
    REQUIRE(theta >= -M_PI - 1e-9);
    REQUIRE(theta <= M_PI + 1e-9);
}

TEST_CASE("Predict covariance grows by process_noise after one step",
          "[predict]") {
    Filter ukf;
    Filter::StateVec x0 = Filter::StateVec::Zero();
    Filter::StateMat P0 = Filter::StateMat::Identity() * 0.5;
    ukf.init(x0, P0);

    ukf.predict(0.1);
    // process_noise(dt) = I * dt placeholder; sigma reconstruction at zero
    // velocity returns ~P0, so final cov ~= P0 + I*dt.
    Filter::StateMat expected = P0 + Filter::StateMat::Identity() * 0.1;
    REQUIRE_THAT(ukf.covariance(), IsApprox(expected, 1e-7));
}
```

Add `immtrack_add_test(test_ukf_predict)` to `tests/cpp/CMakeLists.txt`.

- [ ] **Step 2: Run, verify failures**

```bash
cmake --build build-test --target test_ukf_predict
ctest --test-dir build-test -R test_ukf_predict --output-on-failure
```

Expected: most tests fail (predict is still placeholder mean-only — covariance test fails, theta wrap test fails, linear motion test may pass coincidentally).

- [ ] **Step 3: Implement full predict in `ukf.hpp`**

Replace the body of `UKF::predict` (and remove the placeholder) so the file becomes:

```cpp
#pragma once

#include <stdexcept>

#include <Eigen/Core>

#include <immtrack/detail/sigma_points.hpp>
#include <immtrack/errors.hpp>

namespace immtrack {

template <class Motion, class Obs>
class UKF {
   public:
    static constexpr int N = Motion::N;
    static constexpr int M = Obs::M;
    static constexpr int K = 2 * N + 1;
    using StateVec = Eigen::Matrix<double, N, 1>;
    using StateMat = Eigen::Matrix<double, N, N>;
    using MeasVec = Eigen::Matrix<double, M, 1>;

    UKF() : UKF(1e-3, 2.0, 0.0) {}

    UKF(double alpha, double beta, double kappa)
        : x_(StateVec::Zero()),
          P_(StateMat::Identity()),
          weights_(detail::UnscentedWeights<N>::make(alpha, beta, kappa)) {}

    void init(const StateVec& x, const StateMat& P) {
        x_ = x;
        P_ = P;
    }

    void predict(double dt) {
        if (dt < 0.0) {
            throw InvalidArgument("UKF::predict: dt must be non-negative");
        }

        // 1. Generate sigma points from current state.
        const auto sigmas =
            detail::generate_sigma_points<N>(x_, P_, weights_.lambda);

        // 2. Propagate each sigma point through Motion::predict.
        Eigen::Matrix<double, N, K> propagated;
        for (int i = 0; i < K; ++i) {
            propagated.col(i) = Motion::predict(sigmas.col(i), dt);
        }

        // 3. Reconstruct mean (motion-aware: handles angular dims).
        const StateVec mu_new =
            Motion::template weighted_mean<K>(
                propagated, weights_.mean_weights);

        // 4. Reconstruct covariance using motion-aware residual.
        StateMat P_new = StateMat::Zero();
        for (int i = 0; i < K; ++i) {
            const StateVec d = Motion::residual(propagated.col(i), mu_new);
            P_new.noalias() += weights_.cov_weights(i) * d * d.transpose();
        }
        P_new += Motion::process_noise(dt);

        // 5. Symmetrize to suppress floating-point drift.
        x_ = mu_new;
        P_ = 0.5 * (P_new + P_new.transpose());
    }

    // Placeholder. Full unscented update lands in Task 7 (next).
    double update(const MeasVec& /*z*/) {
        throw std::runtime_error("UKF::update not implemented yet");
    }

    const StateVec& state() const noexcept { return x_; }
    const StateMat& covariance() const noexcept { return P_; }

   private:
    StateVec x_;
    StateMat P_;
    detail::UnscentedWeights<N> weights_;
};

}  // namespace immtrack
```

- [ ] **Step 4: Run all C++ tests**

```bash
cmake --build build-test --target test_angle test_sigma_points test_traits test_ukf_predict
ctest --test-dir build-test --output-on-failure
```

Expected: all 6 predict cases pass plus prior tests still pass.

- [ ] **Step 5: Run Python smoke tests (regression)**

```bash
rm -rf build/ && uv sync
uv run pytest tests/python -q
```

Expected: all 7 smoke tests pass (predict semantics unchanged from caller's view; the Python `test_predict_advances_position_under_cv` exercises the new sigma-point path).

- [ ] **Step 6: Commit**

```bash
git add cpp/include/immtrack/ukf.hpp \
        tests/cpp/test_ukf_predict.cpp \
        tests/cpp/CMakeLists.txt
git commit -m "feat: full sigma-point UKF predict

Replace placeholder mean-only propagation with the Merwe scaled
unscented transform: generate 2N+1 sigma points, propagate each
through Motion::predict, reconstruct mean via Motion::weighted_mean
and covariance via Motion::residual + Motion::process_noise. Symmetrize
final covariance.

Six new Catch2 cases cover linear-motion equivalence, dt=0 no-op,
negative-dt rejection, PSD preservation over 1000 cycles, theta
wrap across +pi, and covariance growth under process noise."
```

---

## Task 7: UKF::update full implementation with NIS return (TDD)

Replace the throwing placeholder with the unscented update step. `update()` returns NIS (a scalar) for downstream IMM / gating.

**Files:**

- Modify: `cpp/include/immtrack/ukf.hpp`
- Create: `tests/cpp/test_ukf_update.cpp`
- Modify: `tests/cpp/CMakeLists.txt`
- Modify: `tests/python/test_smoke.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/cpp/test_ukf_update.cpp`:

```cpp
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <immtrack/ukf.hpp>

#include "eigen_matchers.hpp"

using Catch::Matchers::WithinAbs;
using immtrack::PosVxyzYawCV;
using immtrack::PosYawObs;
using immtrack::UKF;
using immtrack::test::IsApprox;
using immtrack::test::IsPsd;

using Filter = UKF<PosVxyzYawCV, PosYawObs>;

TEST_CASE("Update with measurement at predicted z leaves state "
          "approximately unchanged and shrinks cov", "[update]") {
    Filter ukf;
    Filter::StateVec x0;
    x0 << 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.5;
    Filter::StateMat P0 = Filter::StateMat::Identity() * 0.5;
    ukf.init(x0, P0);

    Filter::MeasVec z;
    z << 1.0, 2.0, 3.0, 0.5;  // exactly the predicted measurement

    const double trace_before = ukf.covariance().trace();
    const double nis = ukf.update(z);
    const double trace_after = ukf.covariance().trace();

    REQUIRE_THAT(ukf.state(), IsApprox(x0, 1e-9));
    REQUIRE(trace_after < trace_before);  // information added shrinks cov
    REQUIRE(nis >= 0.0);
    REQUIRE_THAT(ukf.covariance(), IsPsd());
}

TEST_CASE("Update theta residual wraps across +/- pi boundary", "[update]") {
    Filter ukf;
    Filter::StateVec x0 = Filter::StateVec::Zero();
    x0(6) = M_PI - 0.05;  // state.theta just below +pi
    Filter::StateMat P0 = Filter::StateMat::Identity() * 0.1;
    ukf.init(x0, P0);

    Filter::MeasVec z;
    z << 0.0, 0.0, 0.0, -M_PI + 0.05;  // measurement just above -pi

    const double theta_before = ukf.state()(6);
    ukf.update(z);
    const double theta_after = ukf.state()(6);

    // True residual is ~+0.1 (cross the seam), not ~-2*pi+0.1.
    // Filter should pull theta slightly toward the measurement, not jump
    // by ~2*pi. Expect movement on the order of 0.05, not -6.2.
    const double delta = theta_after - theta_before;
    REQUIRE(std::abs(delta) < 0.2);
}

TEST_CASE("Update converges to ground truth under noisy measurements",
          "[update]") {
    Filter ukf(1e-3, 2.0, 0.0);
    // Wide initial uncertainty, prior mean wrong.
    Filter::StateVec x0 = Filter::StateVec::Zero();
    Filter::StateMat P0 = Filter::StateMat::Identity() * 5.0;
    ukf.init(x0, P0);

    const Filter::MeasVec z_true =
        (Filter::MeasVec() << 2.0, -1.0, 0.5, 0.3).finished();

    // Feed the same measurement repeatedly (no noise simulation needed
    // for a convergence smoke test).
    for (int i = 0; i < 50; ++i) {
        ukf.predict(0.0);
        ukf.update(z_true);
    }

    // Position state should have collapsed near the measurement.
    REQUIRE_THAT(ukf.state()(0), WithinAbs(2.0, 0.05));
    REQUIRE_THAT(ukf.state()(1), WithinAbs(-1.0, 0.05));
    REQUIRE_THAT(ukf.state()(2), WithinAbs(0.5, 0.05));
    REQUIRE_THAT(ukf.state()(6), WithinAbs(0.3, 0.05));
}

TEST_CASE("Update preserves PSD over many cycles", "[update]") {
    Filter ukf;
    Filter::StateVec x0 = Filter::StateVec::Zero();
    Filter::StateMat P0 = Filter::StateMat::Identity();
    ukf.init(x0, P0);

    Filter::MeasVec z;
    z << 0.5, 0.0, 0.0, 0.0;

    for (int i = 0; i < 200; ++i) {
        ukf.predict(0.05);
        ukf.update(z);
    }

    REQUIRE_THAT(ukf.covariance(), IsPsd());
}

TEST_CASE("NIS is non-negative", "[update]") {
    Filter ukf;
    ukf.init(Filter::StateVec::Zero(), Filter::StateMat::Identity());
    Filter::MeasVec z;
    z << 1.0, 1.0, 1.0, 0.5;
    const double nis = ukf.update(z);
    REQUIRE(nis >= 0.0);
}
```

Add `immtrack_add_test(test_ukf_update)` to `tests/cpp/CMakeLists.txt`.

- [ ] **Step 2: Run, verify failures**

```bash
cmake --build build-test --target test_ukf_update
ctest --test-dir build-test -R test_ukf_update --output-on-failure
```

Expected: all 5 cases fail (every test calls `update`, which currently throws `std::runtime_error`).

- [ ] **Step 3: Implement update in `ukf.hpp`**

Replace the placeholder body of `update()` with the full unscented update. The full file becomes:

```cpp
#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <immtrack/detail/sigma_points.hpp>
#include <immtrack/errors.hpp>

namespace immtrack {

template <class Motion, class Obs>
class UKF {
   public:
    static constexpr int N = Motion::N;
    static constexpr int M = Obs::M;
    static constexpr int K = 2 * N + 1;
    using StateVec = Eigen::Matrix<double, N, 1>;
    using StateMat = Eigen::Matrix<double, N, N>;
    using MeasVec = Eigen::Matrix<double, M, 1>;

    UKF() : UKF(1e-3, 2.0, 0.0) {}

    UKF(double alpha, double beta, double kappa)
        : x_(StateVec::Zero()),
          P_(StateMat::Identity()),
          weights_(detail::UnscentedWeights<N>::make(alpha, beta, kappa)) {}

    void init(const StateVec& x, const StateMat& P) {
        x_ = x;
        P_ = P;
    }

    void predict(double dt) {
        if (dt < 0.0) {
            throw InvalidArgument("UKF::predict: dt must be non-negative");
        }

        const auto sigmas =
            detail::generate_sigma_points<N>(x_, P_, weights_.lambda);

        Eigen::Matrix<double, N, K> propagated;
        for (int i = 0; i < K; ++i) {
            propagated.col(i) = Motion::predict(sigmas.col(i), dt);
        }

        const StateVec mu_new =
            Motion::template weighted_mean<K>(
                propagated, weights_.mean_weights);

        StateMat P_new = StateMat::Zero();
        for (int i = 0; i < K; ++i) {
            const StateVec d = Motion::residual(propagated.col(i), mu_new);
            P_new.noalias() += weights_.cov_weights(i) * d * d.transpose();
        }
        P_new += Motion::process_noise(dt);

        x_ = mu_new;
        P_ = 0.5 * (P_new + P_new.transpose());
    }

    // Returns NIS (normalized innovation squared, scalar). Useful for
    // gating and IMM likelihood computation.
    double update(const MeasVec& z) {
        // 1. Fresh sigma points from current (mu, Sigma).
        const auto sigmas =
            detail::generate_sigma_points<N>(x_, P_, weights_.lambda);

        // 2. Project sigma points through observation model.
        Eigen::Matrix<double, M, K> z_sigmas;
        for (int i = 0; i < K; ++i) {
            z_sigmas.col(i) = Obs::h(sigmas.col(i));
        }

        // 3. Predicted measurement mean (obs-aware).
        const MeasVec z_pred =
            Obs::template weighted_mean<K>(z_sigmas, weights_.mean_weights);

        // 4. Innovation covariance S and cross covariance T.
        using MeasMat = Eigen::Matrix<double, M, M>;
        using CrossMat = Eigen::Matrix<double, N, M>;

        MeasMat S = MeasMat::Zero();
        CrossMat T = CrossMat::Zero();
        for (int i = 0; i < K; ++i) {
            const MeasVec dz = Obs::residual(z_sigmas.col(i), z_pred);
            const StateVec dx = Motion::residual(sigmas.col(i), x_);
            S.noalias() += weights_.cov_weights(i) * dz * dz.transpose();
            T.noalias() += weights_.cov_weights(i) * dx * dz.transpose();
        }
        S += Obs::measurement_noise();

        // 5. Kalman gain via LDLT solve (no explicit inverse).
        Eigen::LDLT<MeasMat> S_ldlt(S);
        if (S_ldlt.info() != Eigen::Success) {
            throw NumericalError(
                "UKF::update: innovation covariance not invertible");
        }
        const CrossMat kalman_gain =
            S_ldlt.solve(T.transpose()).transpose();

        // 6. Innovation (obs-aware residual).
        const MeasVec innovation = Obs::residual(z, z_pred);

        // 7. Posterior mean and covariance.
        x_.noalias() += kalman_gain * innovation;
        StateMat P_post = P_ - kalman_gain * S * kalman_gain.transpose();
        P_ = 0.5 * (P_post + P_post.transpose());

        // 8. NIS for downstream IMM / gating.
        return innovation.dot(S_ldlt.solve(innovation));
    }

    const StateVec& state() const noexcept { return x_; }
    const StateMat& covariance() const noexcept { return P_; }

   private:
    StateVec x_;
    StateMat P_;
    detail::UnscentedWeights<N> weights_;
};

}  // namespace immtrack
```

- [ ] **Step 4: Update Python smoke test to exercise update**

Replace `test_update_not_yet_implemented_in_this_commit` with:

```python
def test_update_returns_nis_and_shrinks_cov() -> None:
    f = UkfPosVxyzYawCV()
    x0 = np.zeros(7)
    f.init(state=x0, cov=np.eye(7))

    trace_before = np.trace(f.covariance)
    nis = f.update(measurement=np.zeros(4))
    trace_after = np.trace(f.covariance)

    assert nis >= 0.0
    assert trace_after < trace_before
```

Place it where the previous "not yet implemented" test was in `tests/python/test_smoke.py`.

- [ ] **Step 5: Run all tests**

```bash
cmake --build build-test
ctest --test-dir build-test --output-on-failure

rm -rf build/ && uv sync
uv run pytest tests/python -q
```

Expected:

- C++: 5 test_ukf_update cases pass + all prior tests still pass.
- Python: 7 smoke tests pass.

- [ ] **Step 6: Commit**

```bash
git add cpp/include/immtrack/ukf.hpp \
        tests/cpp/test_ukf_update.cpp \
        tests/cpp/CMakeLists.txt \
        tests/python/test_smoke.py
git commit -m "feat: full sigma-point UKF update returning NIS

Replace throwing placeholder with the unscented update step:
- generate fresh sigma points from current covariance,
- project through Obs::h,
- compute innovation cov S and cross cov T using Obs::residual /
  Motion::residual (theta-wrap aware),
- solve K = T*S^-1 via LDLT (no explicit inverse),
- update mean and symmetrize covariance,
- return NIS = innovation^T * S^-1 * innovation for downstream IMM.

Five new Catch2 cases cover zero-innovation behaviour, theta-wrap
residuals across +/-pi, convergence under repeated measurements,
PSD preservation over 200 predict/update cycles, and non-negative
NIS. Python smoke test now exercises update."
```

---

## Task 8: Final integration verification + README example

End-to-end check that the install path, Python API, and exception mapping all work as documented in the spec. No new logic — this task is purely the verification gate.

**Files:**

- Modify: `cpp/include/immtrack/ukf.hpp` (header docstring only)
- Modify: `tests/python/test_smoke.py` (add exception mapping test)

- [ ] **Step 1: Add Python test for exception mapping**

Append to `tests/python/test_smoke.py`:

```python
def test_predict_with_nonpsd_cov_raises_covariance_not_psd() -> None:
    f = UkfPosVxyzYawCV()
    f.init(state=np.zeros(7), cov=-np.eye(7))
    with pytest.raises(CovarianceNotPsd):
        f.predict(dt=0.1)


def test_invalid_argument_is_value_error() -> None:
    f = UkfPosVxyzYawCV()
    f.init(state=np.zeros(7), cov=np.eye(7))
    with pytest.raises(ValueError):  # InvalidArgument is registered as ValueError
        f.predict(dt=-1.0)
```

- [ ] **Step 2: Add header docstring summarizing the trait contract**

Modify the comment block at the top of `cpp/include/immtrack/ukf.hpp` (just above the `template <class Motion, class Obs>` line). Replace any existing comment with:

```cpp
// Unscented Kalman Filter parameterised by Motion and Obs traits.
//
// Trait contract (compile-time):
//
//   class Motion {
//       static constexpr int N;            // state dimension
//       using State = Eigen::Matrix<double, N, 1>;
//       using Cov   = Eigen::Matrix<double, N, N>;
//
//       static State predict(const State& x, double dt);
//       static Cov   process_noise(double dt);
//
//       template <int K>                                   // K = 2N+1
//       static State weighted_mean(
//           const Eigen::Matrix<double, N, K>& sigmas,
//           const Eigen::Matrix<double, K, 1>& weights);
//
//       static State residual(const State& a, const State& b);
//   };
//
//   class Obs {
//       static constexpr int M;            // measurement dimension
//       using Meas  = Eigen::Matrix<double, M, 1>;
//       using Noise = Eigen::Matrix<double, M, M>;
//
//       template <class State> static Meas h(const State& x);
//       static Noise measurement_noise();
//
//       template <int K>
//       static Meas weighted_mean(
//           const Eigen::Matrix<double, M, K>& sigmas,
//           const Eigen::Matrix<double, K, 1>& weights);
//
//       static Meas residual(const Meas& a, const Meas& b);
//   };
//
// Default constructor: alpha=1e-3, beta=2, kappa=0 (Merwe scaled,
// Gaussian-prior recommended).
// Default state: mu = 0, Sigma = I. Call init() before predict/update
// if a non-identity prior is needed.
//
// Filter::update(z) returns NIS (normalized innovation squared) for
// downstream gating / IMM likelihood.
```

- [ ] **Step 3: Run the full test suite**

```bash
# C++
cmake --build build-test
ctest --test-dir build-test --output-on-failure --schedule-random

# Python
rm -rf build/ && uv sync
uv run pytest tests/python -v
```

Expected: every C++ test case passes, all Python tests pass.

- [ ] **Step 4: Verify the documented Python usage example actually runs**

Run the spec's example as an inline check:

```bash
uv run python <<'PY'
import numpy as np
from immtrack import UkfPosVxyzYawCV

ukf = UkfPosVxyzYawCV()
x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
P0 = np.eye(7) * 0.5
ukf.init(state=x0, cov=P0)

ukf.predict(dt=0.1)
nis = ukf.update(measurement=np.array([0.1, 0.0, 0.0, 0.0]))

print("state =", ukf.state)
print("cov diag =", np.diag(ukf.covariance))
print("NIS =", nis)
print("N =", UkfPosVxyzYawCV.N, "M =", UkfPosVxyzYawCV.M)
PY
```

Expected: prints state ~= [0.1*something_small, 0, 0, ~1.0, 0, 0, 0], NIS >= 0, N = 7, M = 4. No exceptions.

- [ ] **Step 5: Commit**

```bash
git add cpp/include/immtrack/ukf.hpp tests/python/test_smoke.py
git commit -m "test: add exception-mapping smoke tests + ukf.hpp trait docstring

Verify that Python sees CovarianceNotPsd / InvalidArgument (== ValueError)
exactly as the spec promises, and document the Motion / Obs trait
contract in the UKF header so future contributors do not have to
reverse-engineer it from the implementation."
```

---

## Out-of-Scope Reminders

These are explicitly NOT in this plan (deferred to later specs):

- IMM combination of multiple motion models.
- EKF / Particle Filter back-ends.
- Tuned process / measurement noise (placeholders identity-scaled).
- Track lifecycle, data association, gating logic above the filter.
- Filter state serialization.
- Performance benchmarking / SIMD beyond Eigen defaults.
