# BBoxTracker + AMOTA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a multi-object tracker over 3D bounding boxes (`BBoxTracker`) wrapping the existing `UkfPosVxyzYawCV`, with strict per-class association via Hungarian + Mahalanobis gating, M-of-N lifecycle, and EMA size smoothing. Implement AMOTA evaluation in C++, exposed to Python.

**Architecture:** Header-only C++ templates parameterised on `Filter` and `CostPolicy`. Hungarian solver hardcoded. AMOTA reuses the same solver. Single pybind11 module `immtrack._core` exposes `BBoxTracker`, `BoundingBox`, `TrackedObject`, `AmotaConfig`, `AmotaResult`, `MatchMetric`, `amota()`.

**Tech Stack:** C++20, Eigen 3.4, Catch2 v3.5.4 (test), pybind11 (bindings), CMake + Ninja, scikit-build-core, pytest (Python tests).

**Spec:** `docs/superpowers/specs/2026-05-04-bbox-tracker-design.md`

---

## File Structure

```
cpp/include/immtrack/
├── ukf.hpp                  # MODIFY: add predict_measurement()
├── bbox.hpp                 # CREATE: BoundingBox struct
├── tracked_object.hpp       # CREATE: TrackedObject struct
├── tracker.hpp              # CREATE: Track<F>, BBoxTracker<F, Cost>
├── cost_policies.hpp        # CREATE: MahalanobisCost<F>
├── metrics/
│   └── amota.hpp            # CREATE: AmotaConfig, AmotaResult, amota()
└── detail/
    ├── hungarian.hpp        # CREATE: O(n^3) Hungarian solver
    └── iou3d.hpp            # CREATE: 3D IoU helper

bindings/
└── _core.cc                 # MODIFY: export new types and functions

src/immtrack/
├── __init__.py              # MODIFY: re-export new names
├── _core.pyi                # MODIFY: add stubs
└── metrics/
    └── __init__.py          # CREATE: re-export amota / configs

tests/cpp/
├── CMakeLists.txt           # MODIFY: register new tests
├── test_ukf_predict_measurement.cpp  # CREATE
├── test_hungarian.cpp                # CREATE
├── test_mahalanobis_cost.cpp         # CREATE
├── test_track_lifecycle.cpp          # CREATE
├── test_bbox_tracker.cpp             # CREATE
├── test_iou3d.cpp                    # CREATE
└── test_amota.cpp                    # CREATE

tests/python/
├── test_bbox_tracker.py     # CREATE
└── test_amota.py            # CREATE
```

---

## Task 1: Add `predict_measurement()` to UKF

The Mahalanobis cost needs `(z_pred, S)` without consuming the measurement. Add a side-effect-free method to `UKF`.

**Files:**

- Modify: `cpp/include/immtrack/ukf.hpp`
- Create: `tests/cpp/test_ukf_predict_measurement.cpp`
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create `tests/cpp/test_ukf_predict_measurement.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <immtrack/ukf.hpp>

using Catch::Matchers::WithinAbs;

using Ukf = immtrack::UKF<immtrack::PosVxyzYawCV, immtrack::PosYawObs>;

TEST_CASE("predict_measurement returns z_pred and S without mutating state",
          "[ukf][predict_measurement]") {
    Ukf ukf;
    Ukf::StateVec x;
    x << 1.0, 2.0, 3.0, 0.5, 0.0, 0.0, 0.1;
    Ukf::StateMat P = Ukf::StateMat::Identity() * 0.5;
    ukf.init(x, P);

    const auto state_before = ukf.state();
    const auto cov_before = ukf.covariance();

    const auto pm = ukf.predict_measurement();

    REQUIRE(pm.z_pred.size() == 4);
    REQUIRE(pm.S.rows() == 4);
    REQUIRE(pm.S.cols() == 4);

    REQUIRE_THAT(pm.z_pred(0), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(pm.z_pred(1), WithinAbs(2.0, 1e-9));
    REQUIRE_THAT(pm.z_pred(2), WithinAbs(3.0, 1e-9));
    REQUIRE_THAT(pm.z_pred(3), WithinAbs(0.1, 1e-9));

    // S must be positive definite and include measurement noise.
    Eigen::LDLT<Eigen::Matrix<double, 4, 4>> ldlt(pm.S);
    REQUIRE(ldlt.info() == Eigen::Success);
    REQUIRE(pm.S.isApprox(pm.S.transpose(), 1e-12));

    // State and covariance unchanged.
    REQUIRE(ukf.state().isApprox(state_before, 1e-15));
    REQUIRE(ukf.covariance().isApprox(cov_before, 1e-15));
}
```

- [ ] **Step 2: Register the test**

In `tests/cpp/CMakeLists.txt`, add after the last `immtrack_add_test(test_ukf_update)` line:

```cmake
immtrack_add_test(test_ukf_predict_measurement)
```

- [ ] **Step 3: Run the test, expect compile failure**

Run: `cmake -S . -B build-test -G Ninja -DIMMTRACK_BUILD_TESTS=ON && cmake --build build-test`
Expected: build fails with "no member named 'predict_measurement' in 'immtrack::UKF<...>'".

- [ ] **Step 4: Implement `predict_measurement` in UKF**

In `cpp/include/immtrack/ukf.hpp`, inside `class UKF`, add public type and method **after** the `update(...)` method definition (around line 140) and **before** `state()`:

```cpp
    struct PredictedMeasurement {
        MeasVec z_pred;
        Eigen::Matrix<double, M, M> S;
    };

    // Compute predicted measurement and innovation covariance without
    // mutating state. Used for gating / cost computation in tracking.
    PredictedMeasurement predict_measurement() const {
        const auto sigmas =
            detail::generate_sigma_points<N>(x_, P_, weights_.lambda);

        Eigen::Matrix<double, M, K> z_sigmas;
        for (int i = 0; i < K; ++i) {
            z_sigmas.col(i) = Obs::h(sigmas.col(i));
        }

        const MeasVec z_pred =
            Obs::template weighted_mean<K>(z_sigmas, weights_.mean_weights);

        using MeasMat = Eigen::Matrix<double, M, M>;
        MeasMat S = MeasMat::Zero();
        for (int i = 0; i < K; ++i) {
            const MeasVec dz = Obs::residual(z_sigmas.col(i), z_pred);
            S.noalias() += weights_.cov_weights(i) * dz * dz.transpose();
        }
        S += Obs::measurement_noise();
        S = 0.5 * (S + S.transpose());

        return PredictedMeasurement{z_pred, S};
    }
```

- [ ] **Step 5: Run the test, expect pass**

Run: `cmake --build build-test && ctest --test-dir build-test --output-on-failure -R test_ukf_predict_measurement`
Expected: 1/1 test passing.

- [ ] **Step 6: Run the entire C++ suite to ensure no regression**

Run: `ctest --test-dir build-test --output-on-failure`
Expected: all previously-passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add cpp/include/immtrack/ukf.hpp tests/cpp/test_ukf_predict_measurement.cpp tests/cpp/CMakeLists.txt
git commit -m "feat(ukf): add predict_measurement() for non-destructive innovation covariance access"
```

---

## Task 2: Create `BoundingBox` struct

**Files:**

- Create: `cpp/include/immtrack/bbox.hpp`

- [ ] **Step 1: Create `bbox.hpp`**

```cpp
#pragma once

#include <string>

namespace immtrack {

// 3D bounding box used for both detector input and AMOTA evaluation.
// At detection time `track_id` is left at -1; for AMOTA evaluation the
// caller fills it with the GT track ID (for ground truth) or the
// tracker-assigned ID (for predictions).
struct BoundingBox {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double l = 0.0;
    double w = 0.0;
    double h = 0.0;
    double rot = 0.0;
    std::string class_name;
    double score = 0.0;
    int track_id = -1;
};

}  // namespace immtrack
```

- [ ] **Step 2: Smoke build**

Run: `cmake --build build-test`
Expected: build succeeds (header is unused but should compile when included).

- [ ] **Step 3: Commit**

```bash
git add cpp/include/immtrack/bbox.hpp
git commit -m "feat(bbox): add BoundingBox struct (detector input + eval input)"
```

---

## Task 3: Create `TrackedObject` struct

**Files:**

- Create: `cpp/include/immtrack/tracked_object.hpp`

- [ ] **Step 1: Create `tracked_object.hpp`**

```cpp
#pragma once

#include <string>

namespace immtrack {

// Per-track snapshot returned from BBoxTracker::update().
struct TrackedObject {
    int id = -1;
    std::string class_name;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
    double rot = 0.0;
    double l = 0.0;
    double w = 0.0;
    double h = 0.0;
    double score = 0.0;
    int age = 0;
    int hit_count = 0;
    int miss_count = 0;
};

}  // namespace immtrack
```

- [ ] **Step 2: Smoke build**

Run: `cmake --build build-test`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add cpp/include/immtrack/tracked_object.hpp
git commit -m "feat(track): add TrackedObject snapshot struct"
```

---

## Task 4: Hungarian solver

Implement an O(n³) rectangular Hungarian assignment solver.

**Files:**

- Create: `cpp/include/immtrack/detail/hungarian.hpp`
- Create: `tests/cpp/test_hungarian.cpp`
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create `tests/cpp/test_hungarian.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Core>
#include <algorithm>
#include <vector>

#include <immtrack/detail/hungarian.hpp>

using Catch::Matchers::WithinAbs;

namespace {

double total_cost(const Eigen::MatrixXd& cost,
                  const std::vector<std::pair<int, int>>& assignment) {
    double total = 0.0;
    for (const auto& [r, c] : assignment) {
        total += cost(r, c);
    }
    return total;
}

}  // namespace

TEST_CASE("Hungarian: empty matrix returns empty assignment", "[hungarian]") {
    Eigen::MatrixXd cost(0, 0);
    const auto a = immtrack::detail::hungarian(cost);
    REQUIRE(a.empty());
}

TEST_CASE("Hungarian: 1x1 returns single pair", "[hungarian]") {
    Eigen::MatrixXd cost(1, 1);
    cost << 3.0;
    const auto a = immtrack::detail::hungarian(cost);
    REQUIRE(a.size() == 1);
    REQUIRE(a[0].first == 0);
    REQUIRE(a[0].second == 0);
}

TEST_CASE("Hungarian: 3x3 known optimal", "[hungarian]") {
    Eigen::MatrixXd cost(3, 3);
    cost << 4.0, 1.0, 3.0,
            2.0, 0.0, 5.0,
            3.0, 2.0, 2.0;
    // Optimal assignment: (0,1), (1,0), (2,2) -> total 1 + 2 + 2 = 5
    const auto a = immtrack::detail::hungarian(cost);
    REQUIRE(a.size() == 3);
    REQUIRE_THAT(total_cost(cost, a), WithinAbs(5.0, 1e-9));
}

TEST_CASE("Hungarian: rectangular 2x3", "[hungarian]") {
    Eigen::MatrixXd cost(2, 3);
    cost << 1.0, 4.0, 5.0,
            7.0, 2.0, 3.0;
    // Optimal: (0,0)=1, (1,1)=2 -> total 3
    const auto a = immtrack::detail::hungarian(cost);
    REQUIRE(a.size() == 2);
    REQUIRE_THAT(total_cost(cost, a), WithinAbs(3.0, 1e-9));
    // Each row appears at most once.
    std::vector<int> rows;
    for (const auto& [r, c] : a) rows.push_back(r);
    std::sort(rows.begin(), rows.end());
    REQUIRE(rows == std::vector<int>{0, 1});
}

TEST_CASE("Hungarian: rectangular 3x2", "[hungarian]") {
    Eigen::MatrixXd cost(3, 2);
    cost << 1.0, 4.0,
            7.0, 2.0,
            5.0, 6.0;
    // Optimal: (0,0)=1, (1,1)=2 -> total 3 (row 2 unmatched)
    const auto a = immtrack::detail::hungarian(cost);
    REQUIRE(a.size() == 2);
    REQUIRE_THAT(total_cost(cost, a), WithinAbs(3.0, 1e-9));
}

TEST_CASE("Hungarian: all cells infeasible still returns valid pairs",
          "[hungarian]") {
    constexpr double INF = 1e9;
    Eigen::MatrixXd cost(2, 2);
    cost << INF, INF,
            INF, INF;
    const auto a = immtrack::detail::hungarian(cost);
    REQUIRE(a.size() == 2);
    // Caller is responsible for filtering >= INF pairs.
}
```

- [ ] **Step 2: Register the test**

In `tests/cpp/CMakeLists.txt`, add:

```cmake
immtrack_add_test(test_hungarian)
```

- [ ] **Step 3: Run the test, expect compile failure**

Run: `cmake --build build-test`
Expected: build fails — `hungarian.hpp` not found.

- [ ] **Step 4: Implement Hungarian solver**

Create `cpp/include/immtrack/detail/hungarian.hpp`:

```cpp
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace immtrack::detail {

// Rectangular Hungarian (Jonker-Volgenant style) minimum-cost assignment.
// Returns vector of (row, col) pairs of size = min(rows, cols).
//
// Complexity: O(n^3) where n = max(rows, cols) after square padding.
// Padding fill value is 0.0 (caller should ensure real costs are finite
// and non-negative for the unpadded portion; INFEASIBLE cells should use
// a large constant and be filtered post-call).
std::vector<std::pair<int, int>> hungarian(const Eigen::MatrixXd& cost);

inline std::vector<std::pair<int, int>> hungarian(
    const Eigen::MatrixXd& cost) {
    const int rows = static_cast<int>(cost.rows());
    const int cols = static_cast<int>(cost.cols());
    if (rows == 0 || cols == 0) {
        return {};
    }

    const int n = std::max(rows, cols);
    // Pad to n x n with 0.0 in slack cells.
    std::vector<std::vector<double>> a(n + 1,
                                        std::vector<double>(n + 1, 0.0));
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (i <= rows && j <= cols) {
                a[i][j] = cost(i - 1, j - 1);
            }
        }
    }

    constexpr double INF = std::numeric_limits<double>::infinity();
    std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, INF);
        std::vector<char> used(n + 1, false);
        do {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = INF;
            int j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    const double cur = a[i0][j] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> ans(n + 1, -1);
    for (int j = 1; j <= n; ++j) {
        if (p[j] != 0) {
            ans[p[j]] = j;
        }
    }

    std::vector<std::pair<int, int>> result;
    result.reserve(std::min(rows, cols));
    for (int i = 1; i <= rows; ++i) {
        const int j = ans[i];
        if (j >= 1 && j <= cols) {
            result.emplace_back(i - 1, j - 1);
        }
    }
    return result;
}

}  // namespace immtrack::detail
```

- [ ] **Step 5: Run the test, expect pass**

Run: `cmake --build build-test && ctest --test-dir build-test --output-on-failure -R test_hungarian`
Expected: all 6 sections pass.

- [ ] **Step 6: Commit**

```bash
git add cpp/include/immtrack/detail/hungarian.hpp tests/cpp/test_hungarian.cpp tests/cpp/CMakeLists.txt
git commit -m "feat(detail): add Hungarian assignment solver (rectangular, O(n^3))"
```

---

## Task 5: `MahalanobisCost` policy

**Files:**

- Create: `cpp/include/immtrack/cost_policies.hpp`
- Create: `tests/cpp/test_mahalanobis_cost.cpp`
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create `tests/cpp/test_mahalanobis_cost.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <immtrack/bbox.hpp>
#include <immtrack/cost_policies.hpp>
#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <immtrack/ukf.hpp>

using Catch::Matchers::WithinAbs;

using Ukf = immtrack::UKF<immtrack::PosVxyzYawCV, immtrack::PosYawObs>;
using Cost = immtrack::MahalanobisCost<Ukf>;

namespace {

Ukf make_ukf_at(double x, double y, double z, double yaw,
                double cov_diag = 1.0) {
    Ukf ukf;
    Ukf::StateVec s;
    s << x, y, z, 0.0, 0.0, 0.0, yaw;
    ukf.init(s, Ukf::StateMat::Identity() * cov_diag);
    return ukf;
}

immtrack::BoundingBox make_box(double x, double y, double z, double yaw) {
    immtrack::BoundingBox b;
    b.x = x; b.y = y; b.z = z; b.rot = yaw;
    b.l = 4.0; b.w = 2.0; b.h = 1.5;
    b.class_name = "car";
    b.score = 0.9;
    return b;
}

}  // namespace

TEST_CASE("MahalanobisCost: zero distance for matching state and observation",
          "[cost][mahalanobis]") {
    const auto ukf = make_ukf_at(1.0, 2.0, 3.0, 0.5);
    const auto box = make_box(1.0, 2.0, 3.0, 0.5);
    const double d2 = Cost::cost(ukf, box);
    REQUIRE_THAT(d2, WithinAbs(0.0, 1e-9));
}

TEST_CASE("MahalanobisCost: positive distance for offset observation",
          "[cost][mahalanobis]") {
    const auto ukf = make_ukf_at(0.0, 0.0, 0.0, 0.0);
    const auto box = make_box(1.0, 1.0, 0.0, 0.0);
    const double d2 = Cost::cost(ukf, box);
    REQUIRE(d2 > 0.0);
}

TEST_CASE("MahalanobisCost: yaw wraps correctly across pi boundary",
          "[cost][mahalanobis]") {
    const auto ukf = make_ukf_at(0.0, 0.0, 0.0, 3.13);  // close to +pi
    const auto box = make_box(0.0, 0.0, 0.0, -3.13);    // close to -pi
    const double d2 = Cost::cost(ukf, box);
    // Yaw difference after wrapping is small (~0.023 rad), so cost is small.
    REQUIRE(d2 < 0.1);
}

TEST_CASE("MahalanobisCost: gate threshold matches chi^2(4, 0.99)",
          "[cost][mahalanobis]") {
    REQUIRE_THAT(Cost::gate_threshold(), WithinAbs(13.28, 1e-6));
}

TEST_CASE("MahalanobisCost: distance grows with observation offset",
          "[cost][mahalanobis]") {
    const auto ukf = make_ukf_at(0.0, 0.0, 0.0, 0.0, 0.5);
    const double d_near = Cost::cost(ukf, make_box(0.5, 0.0, 0.0, 0.0));
    const double d_far = Cost::cost(ukf, make_box(5.0, 0.0, 0.0, 0.0));
    REQUIRE(d_far > d_near);
}
```

- [ ] **Step 2: Register the test**

In `tests/cpp/CMakeLists.txt`, add:

```cmake
immtrack_add_test(test_mahalanobis_cost)
```

- [ ] **Step 3: Run the test, expect compile failure**

Run: `cmake --build build-test`
Expected: build fails — `cost_policies.hpp` not found.

- [ ] **Step 4: Implement `MahalanobisCost`**

Create `cpp/include/immtrack/cost_policies.hpp`:

```cpp
#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <limits>

#include <immtrack/bbox.hpp>

namespace immtrack {

// Mahalanobis distance using the filter's predicted measurement and
// innovation covariance S. Lower is better. Returns +inf if S is not
// numerically invertible.
//
// Filter must expose:
//   - static constexpr int M (measurement dimension, must be 4 for
//     PosYawObs: [x, y, z, yaw])
//   - PredictedMeasurement predict_measurement() const;
//   - typename Obs (with residual() that wraps yaw)
template <class Filter>
struct MahalanobisCost {
    static double cost(const Filter& f, const BoundingBox& d) {
        static_assert(Filter::M == 4,
                      "MahalanobisCost expects 4-D measurement [x, y, z, yaw]");
        const auto pm = f.predict_measurement();

        typename Filter::MeasVec z;
        z << d.x, d.y, d.z, d.rot;

        // Use observation residual to handle yaw wrapping.
        const typename Filter::MeasVec nu =
            std::remove_reference_t<decltype(pm)>{}.z_pred.size(),
            (void)0,
            Filter::observation_residual(z, pm.z_pred);

        Eigen::LDLT<Eigen::Matrix<double, Filter::M, Filter::M>> ldlt(pm.S);
        if (ldlt.info() != Eigen::Success) {
            return std::numeric_limits<double>::infinity();
        }
        return nu.dot(ldlt.solve(nu));
    }

    static constexpr double gate_threshold() { return 13.28; }
};

}  // namespace immtrack
```

**Note:** The expression above uses `Filter::observation_residual` which we will add in Step 5 (a small wrapper around `Obs::residual` so callers don't need to access `Obs` directly). Replace the body of `cost()` with the cleaner version below after Step 5:

```cpp
        const typename Filter::MeasVec nu =
            Filter::observation_residual(z, pm.z_pred);
```

(Remove the placeholder `std::remove_reference_t<...>` line — that was just to keep the file compilable as a single edit if needed. Use the clean form.)

- [ ] **Step 5: Add `observation_residual` static helper to UKF**

In `cpp/include/immtrack/ukf.hpp`, inside `class UKF`, add **after** the `PredictedMeasurement` struct definition:

```cpp
    static MeasVec observation_residual(const MeasVec& a, const MeasVec& b) {
        return Obs::residual(a, b);
    }
```

- [ ] **Step 6: Clean up `cost_policies.hpp`**

Replace the body of `MahalanobisCost::cost` with the clean form:

```cpp
    static double cost(const Filter& f, const BoundingBox& d) {
        static_assert(Filter::M == 4,
                      "MahalanobisCost expects 4-D measurement [x, y, z, yaw]");
        const auto pm = f.predict_measurement();

        typename Filter::MeasVec z;
        z << d.x, d.y, d.z, d.rot;

        const typename Filter::MeasVec nu =
            Filter::observation_residual(z, pm.z_pred);

        Eigen::LDLT<Eigen::Matrix<double, Filter::M, Filter::M>> ldlt(pm.S);
        if (ldlt.info() != Eigen::Success) {
            return std::numeric_limits<double>::infinity();
        }
        return nu.dot(ldlt.solve(nu));
    }
```

- [ ] **Step 7: Run the tests, expect pass**

Run: `cmake --build build-test && ctest --test-dir build-test --output-on-failure -R test_mahalanobis_cost`
Expected: all 5 sections pass.

- [ ] **Step 8: Commit**

```bash
git add cpp/include/immtrack/cost_policies.hpp cpp/include/immtrack/ukf.hpp tests/cpp/test_mahalanobis_cost.cpp tests/cpp/CMakeLists.txt
git commit -m "feat(cost): add MahalanobisCost policy with chi^2(4, 0.99) gate"
```

---

## Task 6: `Track<Filter>` lifecycle

**Files:**

- Create: `cpp/include/immtrack/tracker.hpp` (Track portion only)
- Create: `tests/cpp/test_track_lifecycle.cpp`
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create `tests/cpp/test_track_lifecycle.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>

#include <immtrack/bbox.hpp>
#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <immtrack/tracker.hpp>
#include <immtrack/ukf.hpp>

using Ukf = immtrack::UKF<immtrack::PosVxyzYawCV, immtrack::PosYawObs>;
using Track = immtrack::Track<Ukf>;

namespace {

immtrack::BoundingBox make_box(double x, double y, double z, double yaw) {
    immtrack::BoundingBox b;
    b.x = x; b.y = y; b.z = z; b.rot = yaw;
    b.l = 4.0; b.w = 2.0; b.h = 1.5;
    b.class_name = "car";
    b.score = 0.9;
    return b;
}

}  // namespace

TEST_CASE("Track: spawns Tentative with hit_count = 1", "[track]") {
    Track t(7, make_box(1, 2, 3, 0.1));
    REQUIRE(t.id() == 7);
    REQUIRE(t.class_name() == "car");
    REQUIRE(t.status() == Track::Status::Tentative);
    REQUIRE(t.hit_count() == 1);
    REQUIRE(t.miss_count() == 0);
    REQUIRE(t.age() == 0);
}

TEST_CASE("Track: confirmed after n_init=3 hits", "[track]") {
    Track t(1, make_box(0, 0, 0, 0));
    t.predict(0.1);
    t.update(make_box(0.5, 0, 0, 0), /*alpha=*/0.7, /*n_init=*/3);
    REQUIRE(t.status() == Track::Status::Tentative);
    REQUIRE(t.hit_count() == 2);

    t.predict(0.1);
    t.update(make_box(1.0, 0, 0, 0), 0.7, 3);
    REQUIRE(t.status() == Track::Status::Confirmed);
    REQUIRE(t.hit_count() == 3);
}

TEST_CASE("Track: deleted after max_age=5 consecutive misses", "[track]") {
    Track t(1, make_box(0, 0, 0, 0));
    for (int i = 0; i < 5; ++i) {
        t.predict(0.1);
        t.mark_missed(/*max_age=*/5);
        REQUIRE(t.status() != Track::Status::Deleted);
    }
    t.predict(0.1);
    t.mark_missed(5);
    REQUIRE(t.status() == Track::Status::Deleted);
    REQUIRE(t.miss_count() == 6);
}

TEST_CASE("Track: EMA size update with alpha=0.5", "[track][size]") {
    auto initial = make_box(0, 0, 0, 0);
    initial.l = 4.0; initial.w = 2.0; initial.h = 1.5;
    Track t(1, initial);

    t.predict(0.1);
    auto next = make_box(0, 0, 0, 0);
    next.l = 6.0; next.w = 3.0; next.h = 2.5;
    t.update(next, /*alpha=*/0.5, /*n_init=*/3);

    const auto sz = t.size();
    // 0.5 * 6 + 0.5 * 4 = 5
    REQUIRE(sz(0) == Catch::Approx(5.0).epsilon(1e-9));
    REQUIRE(sz(1) == Catch::Approx(2.5).epsilon(1e-9));
    REQUIRE(sz(2) == Catch::Approx(2.0).epsilon(1e-9));
}

TEST_CASE("Track: snapshot fills TrackedObject correctly", "[track][snapshot]") {
    Track t(42, make_box(1, 2, 3, 0.5));
    const auto snap = t.snapshot();
    REQUIRE(snap.id == 42);
    REQUIRE(snap.class_name == "car");
    REQUIRE(snap.x == Catch::Approx(1.0));
    REQUIRE(snap.y == Catch::Approx(2.0));
    REQUIRE(snap.z == Catch::Approx(3.0));
    REQUIRE(snap.rot == Catch::Approx(0.5));
    REQUIRE(snap.l == Catch::Approx(4.0));
    REQUIRE(snap.w == Catch::Approx(2.0));
    REQUIRE(snap.h == Catch::Approx(1.5));
    REQUIRE(snap.score == Catch::Approx(0.9));
    REQUIRE(snap.hit_count == 1);
}
```

Add `#include <catch2/catch_approx.hpp>` at the top.

- [ ] **Step 2: Register the test**

In `tests/cpp/CMakeLists.txt`, add:

```cmake
immtrack_add_test(test_track_lifecycle)
```

- [ ] **Step 3: Run the test, expect compile failure**

Run: `cmake --build build-test`
Expected: build fails — `tracker.hpp` not found.

- [ ] **Step 4: Implement `Track<Filter>`**

Create `cpp/include/immtrack/tracker.hpp`:

```cpp
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <string>

#include <immtrack/bbox.hpp>
#include <immtrack/tracked_object.hpp>

namespace immtrack {

template <class Filter>
class Track {
   public:
    enum class Status { Tentative, Confirmed, Deleted };

    Track(int id, const BoundingBox& d)
        : id_(id),
          class_name_(d.class_name),
          size_(d.l, d.w, d.h),
          score_(d.score) {
        typename Filter::StateVec s;
        s.setZero();
        s(0) = d.x;
        s(1) = d.y;
        s(2) = d.z;
        s(6) = d.rot;
        filter_.init(s, Filter::StateMat::Identity());
    }

    void predict(double dt) {
        filter_.predict(dt);
        ++age_;
    }

    void update(const BoundingBox& d, double size_ema_alpha, int n_init) {
        typename Filter::MeasVec z;
        z << d.x, d.y, d.z, d.rot;
        filter_.update(z);

        // EMA size update.
        const Eigen::Vector3d obs(d.l, d.w, d.h);
        size_ = size_ema_alpha * obs + (1.0 - size_ema_alpha) * size_;

        score_ = d.score;
        ++hit_count_;
        miss_count_ = 0;

        if (status_ == Status::Tentative && hit_count_ >= n_init) {
            status_ = Status::Confirmed;
        }
    }

    void mark_missed(int max_age) {
        ++miss_count_;
        if (miss_count_ > max_age) {
            status_ = Status::Deleted;
        }
    }

    int id() const noexcept { return id_; }
    const std::string& class_name() const noexcept { return class_name_; }
    Status status() const noexcept { return status_; }
    int age() const noexcept { return age_; }
    int hit_count() const noexcept { return hit_count_; }
    int miss_count() const noexcept { return miss_count_; }
    double score() const noexcept { return score_; }
    Eigen::Vector3d size() const noexcept { return size_; }
    const Filter& filter() const noexcept { return filter_; }
    Filter& filter() noexcept { return filter_; }

    TrackedObject snapshot() const {
        const auto& x = filter_.state();
        TrackedObject t;
        t.id = id_;
        t.class_name = class_name_;
        t.x = x(0);
        t.y = x(1);
        t.z = x(2);
        t.vx = x(3);
        t.vy = x(4);
        t.vz = x(5);
        t.rot = x(6);
        t.l = size_(0);
        t.w = size_(1);
        t.h = size_(2);
        t.score = score_;
        t.age = age_;
        t.hit_count = hit_count_;
        t.miss_count = miss_count_;
        return t;
    }

   private:
    Filter filter_;
    int id_;
    std::string class_name_;
    Eigen::Vector3d size_;
    double score_;
    int hit_count_ = 1;
    int miss_count_ = 0;
    int age_ = 0;
    Status status_ = Status::Tentative;
};

}  // namespace immtrack
```

- [ ] **Step 5: Run the tests, expect pass**

Run: `cmake --build build-test && ctest --test-dir build-test --output-on-failure -R test_track_lifecycle`
Expected: all 5 sections pass.

- [ ] **Step 6: Commit**

```bash
git add cpp/include/immtrack/tracker.hpp tests/cpp/test_track_lifecycle.cpp tests/cpp/CMakeLists.txt
git commit -m "feat(tracker): add Track<Filter> lifecycle (Tentative/Confirmed/Deleted)"
```

---

## Task 7: `BBoxTracker<Filter, CostPolicy>`

**Files:**

- Modify: `cpp/include/immtrack/tracker.hpp` (append BBoxTracker)
- Create: `tests/cpp/test_bbox_tracker.cpp`
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create `tests/cpp/test_bbox_tracker.cpp`:

```cpp
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <set>
#include <vector>

#include <immtrack/bbox.hpp>
#include <immtrack/cost_policies.hpp>
#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <immtrack/tracker.hpp>
#include <immtrack/ukf.hpp>

using Ukf = immtrack::UKF<immtrack::PosVxyzYawCV, immtrack::PosYawObs>;
using Tracker = immtrack::BBoxTracker<Ukf, immtrack::MahalanobisCost>;

namespace {

immtrack::BoundingBox box(const std::string& cls,
                           double x, double y, double z, double yaw,
                           double score = 0.9) {
    immtrack::BoundingBox b;
    b.class_name = cls;
    b.x = x; b.y = y; b.z = z; b.rot = yaw;
    b.l = 4.0; b.w = 2.0; b.h = 1.5;
    b.score = score;
    return b;
}

}  // namespace

TEST_CASE("BBoxTracker: first frame returns no confirmed tracks",
          "[tracker]") {
    Tracker tr;
    auto out = tr.update({box("car", 0, 0, 0, 0)});
    // n_init=3 by default, so 1 hit isn't enough to be confirmed.
    REQUIRE(out.empty());
    REQUIRE(tr.track_count() == 1);
}

TEST_CASE("BBoxTracker: single object confirmed after n_init=3 frames",
          "[tracker]") {
    Tracker tr;
    tr.update({box("car", 0, 0, 0, 0)});
    tr.update({box("car", 1, 0, 0, 0)}, /*dt=*/0.1);
    auto out = tr.update({box("car", 2, 0, 0, 0)}, 0.1);
    REQUIRE(out.size() == 1);
    REQUIRE(out[0].class_name == "car");
    REQUIRE(out[0].id >= 0);
}

TEST_CASE("BBoxTracker: per-class strict (cars and pedestrians don't mix)",
          "[tracker][per-class]") {
    Tracker tr;
    // Spawn one car and one pedestrian at the same location for 3 frames.
    for (int i = 0; i < 3; ++i) {
        const double x = i * 0.5;
        if (i == 0) {
            tr.update({box("car", x, 0, 0, 0), box("pedestrian", x, 0, 0, 0)});
        } else {
            tr.update({box("car", x, 0, 0, 0), box("pedestrian", x, 0, 0, 0)},
                      0.1);
        }
    }
    auto out = tr.update({box("car", 1.5, 0, 0, 0),
                          box("pedestrian", 1.5, 0, 0, 0)}, 0.1);
    REQUIRE(out.size() == 2);
    std::set<std::string> classes;
    for (const auto& t : out) classes.insert(t.class_name);
    REQUIRE(classes.count("car") == 1);
    REQUIRE(classes.count("pedestrian") == 1);
}

TEST_CASE("BBoxTracker: occlusion (one missed frame) keeps track alive",
          "[tracker][occlusion]") {
    Tracker tr;
    tr.update({box("car", 0, 0, 0, 0)});
    tr.update({box("car", 1, 0, 0, 0)}, 0.1);
    auto out = tr.update({box("car", 2, 0, 0, 0)}, 0.1);
    REQUIRE(out.size() == 1);
    const int id = out[0].id;

    // Miss for one frame.
    auto missed = tr.update({}, 0.1);
    REQUIRE(missed.empty());

    // Reappear — Hungarian reuses the same track.
    auto resumed = tr.update({box("car", 4, 0, 0, 0)}, 0.1);
    REQUIRE(resumed.size() == 1);
    REQUIRE(resumed[0].id == id);
}

TEST_CASE("BBoxTracker: track deleted after max_age misses", "[tracker]") {
    Tracker::Config cfg;
    cfg.max_age = 2;
    Tracker tr(cfg);
    tr.update({box("car", 0, 0, 0, 0)});
    tr.update({box("car", 1, 0, 0, 0)}, 0.1);
    tr.update({box("car", 2, 0, 0, 0)}, 0.1);

    // 3 consecutive misses (max_age=2 => deleted after 3rd miss).
    tr.update({}, 0.1);
    tr.update({}, 0.1);
    tr.update({}, 0.1);

    auto out = tr.update({box("car", 0, 0, 0, 0)}, 0.1);
    // The reappearance creates a new tentative track (not yet confirmed).
    REQUIRE(out.empty());
    REQUIRE(tr.track_count() == 1);  // old deleted, new tentative
}

TEST_CASE("BBoxTracker: reset clears all tracks and resets ID counter",
          "[tracker][reset]") {
    Tracker tr;
    tr.update({box("car", 0, 0, 0, 0)});
    REQUIRE(tr.track_count() == 1);
    tr.reset();
    REQUIRE(tr.track_count() == 0);
}
```

- [ ] **Step 2: Register the test**

In `tests/cpp/CMakeLists.txt`, add:

```cmake
immtrack_add_test(test_bbox_tracker)
```

- [ ] **Step 3: Run the test, expect compile failure**

Run: `cmake --build build-test`
Expected: build fails — `BBoxTracker` not found.

- [ ] **Step 4: Implement `BBoxTracker`**

Append to `cpp/include/immtrack/tracker.hpp` (after `Track` class, before namespace close):

```cpp
template <class Filter, template <class> class CostPolicy>
class BBoxTracker {
   public:
    struct Config {
        int n_init = 3;
        int max_age = 5;
        int min_hits = 3;
        double size_ema_alpha = 0.7;
    };

    explicit BBoxTracker(Config cfg = {}) : cfg_(cfg) {}

    std::vector<TrackedObject> update(
        const std::vector<BoundingBox>& detections) {
        return update_impl(detections, /*has_dt=*/false, /*dt=*/0.0);
    }

    std::vector<TrackedObject> update(
        const std::vector<BoundingBox>& detections, double dt) {
        return update_impl(detections, /*has_dt=*/true, dt);
    }

    void reset() {
        tracks_.clear();
        next_id_ = 0;
    }

    std::size_t track_count() const noexcept { return tracks_.size(); }

   private:
    using TrackT = Track<Filter>;

    static constexpr double INFEASIBLE = 1e9;

    std::vector<TrackedObject> update_impl(
        const std::vector<BoundingBox>& detections, bool has_dt, double dt) {
        // 1. Predict every non-deleted track.
        if (has_dt) {
            for (auto& t : tracks_) {
                if (t.status() != TrackT::Status::Deleted) {
                    t.predict(dt);
                }
            }
        }

        // 2. Group track indices and detection indices by class_name.
        std::unordered_map<std::string, std::vector<int>> track_by_class;
        std::unordered_map<std::string, std::vector<int>> det_by_class;

        for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
            if (tracks_[i].status() != TrackT::Status::Deleted) {
                track_by_class[tracks_[i].class_name()].push_back(i);
            }
        }
        for (int j = 0; j < static_cast<int>(detections.size()); ++j) {
            det_by_class[detections[j].class_name].push_back(j);
        }

        std::vector<bool> det_matched(detections.size(), false);
        std::vector<bool> track_matched(tracks_.size(), false);

        // 3. Per-class Hungarian.
        for (const auto& [cls, det_idxs] : det_by_class) {
            auto trk_it = track_by_class.find(cls);
            if (trk_it == track_by_class.end() || trk_it->second.empty()) {
                continue;
            }
            const auto& trk_idxs = trk_it->second;

            const int rows = static_cast<int>(trk_idxs.size());
            const int cols = static_cast<int>(det_idxs.size());
            Eigen::MatrixXd cost(rows, cols);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    const double v = CostPolicy<Filter>::cost(
                        tracks_[trk_idxs[r]].filter(),
                        detections[det_idxs[c]]);
                    cost(r, c) = (v >= CostPolicy<Filter>::gate_threshold())
                                     ? INFEASIBLE
                                     : v;
                }
            }

            const auto pairs = detail::hungarian(cost);
            for (const auto& [r, c] : pairs) {
                if (cost(r, c) >= INFEASIBLE) continue;
                const int trk = trk_idxs[r];
                const int det = det_idxs[c];
                tracks_[trk].update(detections[det], cfg_.size_ema_alpha,
                                    cfg_.n_init);
                track_matched[trk] = true;
                det_matched[det] = true;
            }
        }

        // 4. Mark unmatched tracks missed.
        for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
            if (!track_matched[i] &&
                tracks_[i].status() != TrackT::Status::Deleted) {
                tracks_[i].mark_missed(cfg_.max_age);
            }
        }

        // 5. Spawn new tracks for unmatched detections.
        for (int j = 0; j < static_cast<int>(detections.size()); ++j) {
            if (!det_matched[j]) {
                tracks_.emplace_back(next_id_++, detections[j]);
            }
        }

        // 6. Garbage collect deleted tracks.
        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(),
                           [](const TrackT& t) {
                               return t.status() == TrackT::Status::Deleted;
                           }),
            tracks_.end());

        // 7. Output: confirmed + matched-this-frame + hit_count >= min_hits.
        std::vector<TrackedObject> out;
        out.reserve(tracks_.size());
        for (const auto& t : tracks_) {
            if (t.status() == TrackT::Status::Confirmed &&
                t.miss_count() == 0 &&
                t.hit_count() >= cfg_.min_hits) {
                out.push_back(t.snapshot());
            }
        }
        return out;
    }

    Config cfg_;
    std::vector<TrackT> tracks_;
    int next_id_ = 0;
};
```

Also add the required includes at the top of `tracker.hpp`:

```cpp
#include <unordered_map>
#include <vector>

#include <immtrack/cost_policies.hpp>
#include <immtrack/detail/hungarian.hpp>
```

- [ ] **Step 5: Run the tests, expect pass**

Run: `cmake --build build-test && ctest --test-dir build-test --output-on-failure -R test_bbox_tracker`
Expected: all 6 sections pass.

- [ ] **Step 6: Run full suite to ensure no regression**

Run: `ctest --test-dir build-test --output-on-failure`

- [ ] **Step 7: Commit**

```bash
git add cpp/include/immtrack/tracker.hpp tests/cpp/test_bbox_tracker.cpp tests/cpp/CMakeLists.txt
git commit -m "feat(tracker): add BBoxTracker with per-class Hungarian + Mahalanobis gating"
```

---

## Task 8: 3D IoU helper

**Files:**

- Create: `cpp/include/immtrack/detail/iou3d.hpp`
- Create: `tests/cpp/test_iou3d.cpp`
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create `tests/cpp/test_iou3d.cpp`:

```cpp
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <immtrack/bbox.hpp>
#include <immtrack/detail/iou3d.hpp>

using immtrack::detail::iou3d;

namespace {

immtrack::BoundingBox make(double x, double y, double z, double l, double w,
                           double h, double rot = 0.0) {
    immtrack::BoundingBox b;
    b.x = x; b.y = y; b.z = z;
    b.l = l; b.w = w; b.h = h;
    b.rot = rot;
    return b;
}

}  // namespace

TEST_CASE("iou3d: identical boxes give 1.0", "[iou3d]") {
    auto a = make(0, 0, 0, 4, 2, 1.5);
    REQUIRE(iou3d(a, a) == Catch::Approx(1.0).margin(1e-9));
}

TEST_CASE("iou3d: disjoint boxes give 0.0", "[iou3d]") {
    auto a = make(0, 0, 0, 1, 1, 1);
    auto b = make(10, 10, 10, 1, 1, 1);
    REQUIRE(iou3d(a, b) == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("iou3d: half-overlap along x", "[iou3d]") {
    auto a = make(0, 0, 0, 2, 2, 2);   // [-1,1] x [-1,1] x [-1,1]
    auto b = make(1, 0, 0, 2, 2, 2);   // [0,2] x [-1,1] x [-1,1]
    // Intersection: [0,1] x [-1,1] x [-1,1] -> volume 1*2*2 = 4
    // Union: 8 + 8 - 4 = 12
    REQUIRE(iou3d(a, b) == Catch::Approx(4.0 / 12.0).epsilon(1e-9));
}

TEST_CASE("iou3d: rotation changes IoU", "[iou3d]") {
    auto a = make(0, 0, 0, 4, 2, 1.5, 0.0);
    auto b = make(0, 0, 0, 4, 2, 1.5, 1.5707963);  // 90 deg
    const double v = iou3d(a, b);
    REQUIRE(v > 0.0);
    REQUIRE(v < 1.0);
}
```

- [ ] **Step 2: Register the test**

In `tests/cpp/CMakeLists.txt`, add:

```cmake
immtrack_add_test(test_iou3d)
```

- [ ] **Step 3: Run the test, expect compile failure**

- [ ] **Step 4: Implement `iou3d`**

Create `cpp/include/immtrack/detail/iou3d.hpp`:

```cpp
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <immtrack/bbox.hpp>

namespace immtrack::detail {

// Sutherland-Hodgman polygon clipping for two convex polygons in 2D.
// Both inputs are CCW; returns intersection polygon (CCW).
inline std::vector<std::array<double, 2>> sh_clip(
    const std::vector<std::array<double, 2>>& subject,
    const std::vector<std::array<double, 2>>& clip) {
    std::vector<std::array<double, 2>> out = subject;
    for (size_t i = 0; i < clip.size(); ++i) {
        if (out.empty()) break;
        const auto& a = clip[i];
        const auto& b = clip[(i + 1) % clip.size()];
        const double ex = b[0] - a[0];
        const double ey = b[1] - a[1];
        std::vector<std::array<double, 2>> input;
        input.swap(out);
        for (size_t k = 0; k < input.size(); ++k) {
            const auto& p = input[k];
            const auto& q = input[(k + 1) % input.size()];
            const double dp = ex * (p[1] - a[1]) - ey * (p[0] - a[0]);
            const double dq = ex * (q[1] - a[1]) - ey * (q[0] - a[0]);
            const bool p_in = dp >= 0.0;
            const bool q_in = dq >= 0.0;
            if (p_in) out.push_back(p);
            if (p_in != q_in) {
                const double t = dp / (dp - dq);
                out.push_back({p[0] + t * (q[0] - p[0]),
                               p[1] + t * (q[1] - p[1])});
            }
        }
    }
    return out;
}

inline double polygon_area(const std::vector<std::array<double, 2>>& poly) {
    if (poly.size() < 3) return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const auto& p = poly[i];
        const auto& q = poly[(i + 1) % poly.size()];
        s += p[0] * q[1] - q[0] * p[1];
    }
    return std::abs(s) * 0.5;
}

inline std::vector<std::array<double, 2>> bev_corners(const BoundingBox& b) {
    const double c = std::cos(b.rot);
    const double s = std::sin(b.rot);
    const double hl = b.l * 0.5;
    const double hw = b.w * 0.5;
    // Corners in box frame: (+l/2, +w/2), (-l/2, +w/2), (-l/2, -w/2),
    // (+l/2, -w/2). Order CCW.
    const std::array<std::array<double, 2>, 4> local = {{
        {{ hl,  hw}}, {{-hl,  hw}}, {{-hl, -hw}}, {{ hl, -hw}}
    }};
    std::vector<std::array<double, 2>> world(4);
    for (int i = 0; i < 4; ++i) {
        world[i][0] = b.x + c * local[i][0] - s * local[i][1];
        world[i][1] = b.y + s * local[i][0] + c * local[i][1];
    }
    return world;
}

inline double iou3d(const BoundingBox& a, const BoundingBox& b) {
    // BEV intersection area via Sutherland-Hodgman.
    const auto ca = bev_corners(a);
    const auto cb = bev_corners(b);
    const auto inter_poly = sh_clip(ca, cb);
    const double inter_area = polygon_area(inter_poly);

    // Vertical (z) intersection.
    const double a_lo = a.z - a.h * 0.5;
    const double a_hi = a.z + a.h * 0.5;
    const double b_lo = b.z - b.h * 0.5;
    const double b_hi = b.z + b.h * 0.5;
    const double z_inter = std::max(0.0, std::min(a_hi, b_hi) -
                                              std::max(a_lo, b_lo));

    const double inter_vol = inter_area * z_inter;
    const double vol_a = a.l * a.w * a.h;
    const double vol_b = b.l * b.w * b.h;
    const double union_vol = vol_a + vol_b - inter_vol;
    if (union_vol <= 0.0) return 0.0;
    return inter_vol / union_vol;
}

}  // namespace immtrack::detail
```

Add `#include <vector>` to the includes.

- [ ] **Step 5: Run the tests, expect pass**

Run: `cmake --build build-test && ctest --test-dir build-test --output-on-failure -R test_iou3d`

- [ ] **Step 6: Commit**

```bash
git add cpp/include/immtrack/detail/iou3d.hpp tests/cpp/test_iou3d.cpp tests/cpp/CMakeLists.txt
git commit -m "feat(detail): add 3D IoU helper using Sutherland-Hodgman BEV clipping"
```

---

## Task 9: AMOTA function

**Files:**

- Create: `cpp/include/immtrack/metrics/amota.hpp`
- Create: `tests/cpp/test_amota.cpp`
- Modify: `tests/cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create `tests/cpp/test_amota.cpp`:

```cpp
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

#include <immtrack/bbox.hpp>
#include <immtrack/metrics/amota.hpp>

using immtrack::BoundingBox;
using immtrack::metrics::AmotaConfig;
using immtrack::metrics::amota;
using immtrack::metrics::MatchMetric;

namespace {

BoundingBox box(int track_id, double x, double y, double z, double yaw,
                const std::string& cls = "car", double score = 1.0) {
    BoundingBox b;
    b.track_id = track_id;
    b.class_name = cls;
    b.x = x; b.y = y; b.z = z; b.rot = yaw;
    b.l = 4.0; b.w = 2.0; b.h = 1.5;
    b.score = score;
    return b;
}

}  // namespace

TEST_CASE("amota: empty inputs return 0.0 overall", "[amota]") {
    const auto r = amota({}, {}, {});
    REQUIRE(r.overall == Catch::Approx(0.0));
    REQUIRE(r.per_class.empty());
}

TEST_CASE("amota: perfect predictions yield AMOTA = 1.0", "[amota]") {
    std::vector<std::vector<BoundingBox>> gt = {
        {box(1, 0, 0, 0, 0)},
        {box(1, 1, 0, 0, 0)},
        {box(1, 2, 0, 0, 0)},
    };
    std::vector<std::vector<BoundingBox>> pred = {
        {box(100, 0, 0, 0, 0, "car", 0.9)},
        {box(100, 1, 0, 0, 0, "car", 0.9)},
        {box(100, 2, 0, 0, 0, "car", 0.9)},
    };
    const auto r = amota(gt, pred, {});
    REQUIRE(r.overall == Catch::Approx(1.0).margin(1e-9));
    REQUIRE(r.per_class.at("car") == Catch::Approx(1.0).margin(1e-9));
}

TEST_CASE("amota: pure FP predictions degrade AMOTA", "[amota]") {
    std::vector<std::vector<BoundingBox>> gt = {
        {box(1, 0, 0, 0, 0)},
        {box(1, 1, 0, 0, 0)},
    };
    std::vector<std::vector<BoundingBox>> pred = {
        {box(100, 100, 100, 100, 0, "car", 0.9)},  // far FP
        {box(100, 200, 200, 200, 0, "car", 0.9)},  // far FP
    };
    const auto r = amota(gt, pred, {});
    REQUIRE(r.per_class.at("car") < 1.0);
}

TEST_CASE("amota: ID switch is counted", "[amota]") {
    std::vector<std::vector<BoundingBox>> gt = {
        {box(1, 0, 0, 0, 0)},
        {box(1, 1, 0, 0, 0)},
        {box(1, 2, 0, 0, 0)},
    };
    std::vector<std::vector<BoundingBox>> pred = {
        {box(100, 0, 0, 0, 0, "car", 0.9)},
        {box(200, 1, 0, 0, 0, "car", 0.9)},  // ID switch!
        {box(200, 2, 0, 0, 0, "car", 0.9)},
    };
    const auto r = amota(gt, pred, {});
    // 1 IDS reduces MOTA at every recall threshold.
    REQUIRE(r.per_class.at("car") < 1.0);
}

TEST_CASE("amota: per-class breakdown for mixed classes", "[amota]") {
    std::vector<std::vector<BoundingBox>> gt = {
        {box(1, 0, 0, 0, 0, "car"), box(2, 5, 5, 0, 0, "pedestrian")},
        {box(1, 1, 0, 0, 0, "car"), box(2, 5, 6, 0, 0, "pedestrian")},
    };
    std::vector<std::vector<BoundingBox>> pred = {
        {box(10, 0, 0, 0, 0, "car", 0.9),
         box(20, 5, 5, 0, 0, "pedestrian", 0.9)},
        {box(10, 1, 0, 0, 0, "car", 0.9),
         box(20, 5, 6, 0, 0, "pedestrian", 0.9)},
    };
    const auto r = amota(gt, pred, {});
    REQUIRE(r.per_class.at("car") == Catch::Approx(1.0).margin(1e-9));
    REQUIRE(r.per_class.at("pedestrian") == Catch::Approx(1.0).margin(1e-9));
}

TEST_CASE("amota: Iou3d match metric", "[amota][iou3d]") {
    AmotaConfig cfg;
    cfg.metric = MatchMetric::Iou3d;
    cfg.match_threshold = 0.5;  // need IoU > 0.5

    std::vector<std::vector<BoundingBox>> gt = {
        {box(1, 0, 0, 0, 0)},
    };
    std::vector<std::vector<BoundingBox>> pred = {
        {box(100, 0.1, 0, 0, 0, "car", 0.9)},
    };
    const auto r = amota(gt, pred, cfg);
    REQUIRE(r.per_class.at("car") == Catch::Approx(1.0).margin(1e-9));
}
```

- [ ] **Step 2: Register the test**

In `tests/cpp/CMakeLists.txt`, add:

```cmake
immtrack_add_test(test_amota)
```

- [ ] **Step 3: Run the test, expect compile failure**

- [ ] **Step 4: Implement `amota`**

Create `cpp/include/immtrack/metrics/amota.hpp`:

```cpp
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <immtrack/bbox.hpp>
#include <immtrack/detail/hungarian.hpp>
#include <immtrack/detail/iou3d.hpp>

namespace immtrack::metrics {

enum class MatchMetric { CenterDistance, Iou3d };

struct AmotaConfig {
    std::vector<double> recall_values = {0.1, 0.2, 0.3, 0.4, 0.5,
                                          0.6, 0.7, 0.8, 0.9};
    MatchMetric metric = MatchMetric::CenterDistance;
    double match_threshold = 2.0;  // metres for CenterDistance,
                                   // (1 - IoU) for Iou3d
};

struct AmotaResult {
    double overall = 0.0;
    std::unordered_map<std::string, double> per_class;
};

namespace detail_amota {

inline double pair_cost(const BoundingBox& a, const BoundingBox& b,
                        MatchMetric metric) {
    if (metric == MatchMetric::CenterDistance) {
        const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    return 1.0 - immtrack::detail::iou3d(a, b);
}

// For a fixed score cutoff, run frame-by-frame Hungarian and accumulate
// FP, FN, IDS for the given class. Returns (FP, FN, IDS, gt_count).
struct CountResult {
    int fp = 0;
    int fn = 0;
    int ids = 0;
    int gt_count = 0;
};

inline CountResult count_for_cutoff(
    const std::vector<std::vector<BoundingBox>>& gt_frames,
    const std::vector<std::vector<BoundingBox>>& pred_frames,
    const std::string& cls, double score_cutoff,
    MatchMetric metric, double match_threshold) {
    constexpr double INFEASIBLE = 1e9;
    CountResult r;

    // Maps GT track_id -> last matched pred track_id (for IDS).
    std::unordered_map<int, int> prev_match;

    const std::size_t F = gt_frames.size();
    for (std::size_t f = 0; f < F; ++f) {
        std::vector<const BoundingBox*> gts;
        for (const auto& g : gt_frames[f]) {
            if (g.class_name == cls) gts.push_back(&g);
        }
        std::vector<const BoundingBox*> preds;
        for (const auto& p :
             (f < pred_frames.size() ? pred_frames[f]
                                     : std::vector<BoundingBox>{})) {
            if (p.class_name == cls && p.score >= score_cutoff) {
                preds.push_back(&p);
            }
        }
        r.gt_count += static_cast<int>(gts.size());

        if (gts.empty()) {
            r.fp += static_cast<int>(preds.size());
            continue;
        }
        if (preds.empty()) {
            r.fn += static_cast<int>(gts.size());
            continue;
        }

        Eigen::MatrixXd cost(gts.size(), preds.size());
        for (std::size_t i = 0; i < gts.size(); ++i) {
            for (std::size_t j = 0; j < preds.size(); ++j) {
                const double c = pair_cost(*gts[i], *preds[j], metric);
                cost(static_cast<int>(i), static_cast<int>(j)) =
                    (c > match_threshold) ? INFEASIBLE : c;
            }
        }

        const auto pairs = immtrack::detail::hungarian(cost);
        std::unordered_set<int> matched_gt, matched_pred;
        std::unordered_map<int, int> this_match;

        for (const auto& [i, j] : pairs) {
            if (cost(i, j) >= INFEASIBLE) continue;
            matched_gt.insert(i);
            matched_pred.insert(j);
            const int gt_id = gts[i]->track_id;
            const int pred_id = preds[j]->track_id;
            this_match[gt_id] = pred_id;
            auto it = prev_match.find(gt_id);
            if (it != prev_match.end() && it->second != pred_id) {
                r.ids += 1;
            }
        }

        r.fp += static_cast<int>(preds.size() - matched_pred.size());
        r.fn += static_cast<int>(gts.size() - matched_gt.size());

        prev_match = this_match;
    }
    return r;
}

}  // namespace detail_amota

inline AmotaResult amota(
    const std::vector<std::vector<BoundingBox>>& gt_frames,
    const std::vector<std::vector<BoundingBox>>& pred_frames,
    const AmotaConfig& cfg = {}) {
    AmotaResult result;

    // Collect class set and per-class GT counts.
    std::unordered_set<std::string> classes;
    std::unordered_map<std::string, int> gt_count;
    for (const auto& frame : gt_frames) {
        for (const auto& g : frame) {
            classes.insert(g.class_name);
            gt_count[g.class_name] += 1;
        }
    }
    if (classes.empty() || cfg.recall_values.empty()) {
        return result;
    }

    double weighted_sum = 0.0;
    int total_gt = 0;

    for (const auto& cls : classes) {
        // Collect predictions of this class with their scores; sort desc.
        std::vector<double> scores;
        for (const auto& frame : pred_frames) {
            for (const auto& p : frame) {
                if (p.class_name == cls) scores.push_back(p.score);
            }
        }
        std::sort(scores.begin(), scores.end(), std::greater<double>());

        const int gt_n = gt_count[cls];
        if (gt_n == 0) continue;

        double mota_sum = 0.0;
        for (double r : cfg.recall_values) {
            // Find smallest score cutoff that yields recall >= r.
            // Recall = TP / GT. We brute-force search the unique score
            // values from high to low.
            double best_cutoff = std::numeric_limits<double>::infinity();
            bool found = false;
            std::vector<double> uniq = scores;
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
            // Always include +inf to mean "no predictions".
            for (double cutoff : uniq) {
                const auto cr = detail_amota::count_for_cutoff(
                    gt_frames, pred_frames, cls, cutoff,
                    cfg.metric, cfg.match_threshold);
                const int tp = cr.gt_count - cr.fn;
                const double recall = (cr.gt_count == 0)
                                          ? 0.0
                                          : static_cast<double>(tp) /
                                                cr.gt_count;
                if (recall >= r) {
                    best_cutoff = cutoff;
                    found = true;
                    break;
                }
            }

            double mota = 0.0;
            if (found) {
                const auto cr = detail_amota::count_for_cutoff(
                    gt_frames, pred_frames, cls, best_cutoff,
                    cfg.metric, cfg.match_threshold);
                const double bad = cr.fp + cr.fn + cr.ids;
                const double m = 1.0 - (bad / std::max(1, cr.gt_count));
                mota = std::max(0.0, m);
            }
            mota_sum += mota;
        }

        const double amota_cls =
            mota_sum / static_cast<double>(cfg.recall_values.size());
        result.per_class[cls] = amota_cls;
        weighted_sum += amota_cls * gt_n;
        total_gt += gt_n;
    }

    result.overall =
        (total_gt > 0) ? weighted_sum / static_cast<double>(total_gt) : 0.0;
    return result;
}

}  // namespace immtrack::metrics
```

- [ ] **Step 5: Run the tests, expect pass**

Run: `cmake --build build-test && ctest --test-dir build-test --output-on-failure -R test_amota`

- [ ] **Step 6: Run full suite to ensure no regression**

Run: `ctest --test-dir build-test --output-on-failure`
Expected: all tests still pass.

- [ ] **Step 7: Commit**

```bash
git add cpp/include/immtrack/metrics/amota.hpp tests/cpp/test_amota.cpp tests/cpp/CMakeLists.txt
git commit -m "feat(metrics): add AMOTA evaluation in C++ with center-distance and 3D-IoU matching"
```

---

## Task 10: Pybind11 bindings

Expose `BoundingBox`, `TrackedObject`, `BBoxTracker`, `MatchMetric`, `AmotaConfig`, `AmotaResult`, and `amota()` to Python.

**Files:**

- Modify: `bindings/_core.cc`

- [ ] **Step 1: Read the current bindings file to find the insertion point**

Run: `cat bindings/_core.cc` and identify the `PYBIND11_MODULE(_core, m)` body.

- [ ] **Step 2: Add includes**

At the top of `bindings/_core.cc`, ensure these are present:

```cpp
#include <immtrack/bbox.hpp>
#include <immtrack/cost_policies.hpp>
#include <immtrack/metrics/amota.hpp>
#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <immtrack/tracked_object.hpp>
#include <immtrack/tracker.hpp>
#include <immtrack/ukf.hpp>

#include <pybind11/stl.h>
```

- [ ] **Step 3: Add type bindings**

Inside the `PYBIND11_MODULE(_core, m)` body, append:

```cpp
    namespace py = pybind11;

    py::class_<immtrack::BoundingBox>(m, "BoundingBox")
        .def(py::init<>())
        .def(py::init([](double x, double y, double z, double l, double w,
                          double h, double rot, std::string class_name,
                          double score, int track_id) {
                 immtrack::BoundingBox b;
                 b.x = x; b.y = y; b.z = z;
                 b.l = l; b.w = w; b.h = h;
                 b.rot = rot;
                 b.class_name = std::move(class_name);
                 b.score = score;
                 b.track_id = track_id;
                 return b;
             }),
             py::arg("x") = 0.0, py::arg("y") = 0.0, py::arg("z") = 0.0,
             py::arg("l") = 0.0, py::arg("w") = 0.0, py::arg("h") = 0.0,
             py::arg("rot") = 0.0, py::arg("class_name") = std::string(),
             py::arg("score") = 0.0, py::arg("track_id") = -1)
        .def_readwrite("x", &immtrack::BoundingBox::x)
        .def_readwrite("y", &immtrack::BoundingBox::y)
        .def_readwrite("z", &immtrack::BoundingBox::z)
        .def_readwrite("l", &immtrack::BoundingBox::l)
        .def_readwrite("w", &immtrack::BoundingBox::w)
        .def_readwrite("h", &immtrack::BoundingBox::h)
        .def_readwrite("rot", &immtrack::BoundingBox::rot)
        .def_readwrite("class_name", &immtrack::BoundingBox::class_name)
        .def_readwrite("score", &immtrack::BoundingBox::score)
        .def_readwrite("track_id", &immtrack::BoundingBox::track_id);

    py::class_<immtrack::TrackedObject>(m, "TrackedObject")
        .def_readonly("id", &immtrack::TrackedObject::id)
        .def_readonly("class_name", &immtrack::TrackedObject::class_name)
        .def_readonly("x", &immtrack::TrackedObject::x)
        .def_readonly("y", &immtrack::TrackedObject::y)
        .def_readonly("z", &immtrack::TrackedObject::z)
        .def_readonly("vx", &immtrack::TrackedObject::vx)
        .def_readonly("vy", &immtrack::TrackedObject::vy)
        .def_readonly("vz", &immtrack::TrackedObject::vz)
        .def_readonly("rot", &immtrack::TrackedObject::rot)
        .def_readonly("l", &immtrack::TrackedObject::l)
        .def_readonly("w", &immtrack::TrackedObject::w)
        .def_readonly("h", &immtrack::TrackedObject::h)
        .def_readonly("score", &immtrack::TrackedObject::score)
        .def_readonly("age", &immtrack::TrackedObject::age)
        .def_readonly("hit_count", &immtrack::TrackedObject::hit_count)
        .def_readonly("miss_count", &immtrack::TrackedObject::miss_count);

    using Ukf = immtrack::UKF<immtrack::PosVxyzYawCV, immtrack::PosYawObs>;
    using Tracker = immtrack::BBoxTracker<Ukf, immtrack::MahalanobisCost>;

    py::class_<Tracker> tracker(m, "BBoxTracker");
    tracker
        .def(py::init([](int n_init, int max_age, int min_hits,
                          double size_ema_alpha) {
                 Tracker::Config c;
                 c.n_init = n_init;
                 c.max_age = max_age;
                 c.min_hits = min_hits;
                 c.size_ema_alpha = size_ema_alpha;
                 return Tracker(c);
             }),
             py::arg("n_init") = 3, py::arg("max_age") = 5,
             py::arg("min_hits") = 3, py::arg("size_ema_alpha") = 0.7)
        .def("update",
             py::overload_cast<const std::vector<immtrack::BoundingBox>&>(
                 &Tracker::update),
             py::arg("detections"))
        .def("update",
             py::overload_cast<const std::vector<immtrack::BoundingBox>&,
                                double>(&Tracker::update),
             py::arg("detections"), py::arg("dt"))
        .def("reset", &Tracker::reset)
        .def("track_count", &Tracker::track_count);

    auto metrics = m.def_submodule("metrics", "AMOTA evaluation utilities");

    py::enum_<immtrack::metrics::MatchMetric>(metrics, "MatchMetric")
        .value("CenterDistance",
               immtrack::metrics::MatchMetric::CenterDistance)
        .value("Iou3d", immtrack::metrics::MatchMetric::Iou3d);

    py::class_<immtrack::metrics::AmotaConfig>(metrics, "AmotaConfig")
        .def(py::init<>())
        .def_readwrite("recall_values",
                       &immtrack::metrics::AmotaConfig::recall_values)
        .def_readwrite("metric", &immtrack::metrics::AmotaConfig::metric)
        .def_readwrite("match_threshold",
                       &immtrack::metrics::AmotaConfig::match_threshold);

    py::class_<immtrack::metrics::AmotaResult>(metrics, "AmotaResult")
        .def_readonly("overall", &immtrack::metrics::AmotaResult::overall)
        .def_readonly("per_class",
                      &immtrack::metrics::AmotaResult::per_class);

    metrics.def("amota", &immtrack::metrics::amota,
                py::arg("gt"), py::arg("pred"),
                py::arg("cfg") = immtrack::metrics::AmotaConfig{});
```

- [ ] **Step 4: Rebuild Python extension**

Run: `uv sync --reinstall-package immtrack`
Expected: build succeeds.

- [ ] **Step 5: Smoke test from Python REPL**

Run:

```bash
uv run python -c "
import immtrack
b = immtrack.BoundingBox(x=1, y=2, z=3, l=4, w=2, h=1.5, rot=0, class_name='car', score=0.9)
tr = immtrack.BBoxTracker()
print(tr.update([b]))
print('OK')
"
```

Expected: prints `[]` (n_init=3 not yet reached) and `OK`.

- [ ] **Step 6: Commit**

```bash
git add bindings/_core.cc
git commit -m "feat(bindings): expose BoundingBox, TrackedObject, BBoxTracker, AMOTA to Python"
```

---

## Task 11: Python type stubs and `metrics` package

**Files:**

- Modify: `src/immtrack/_core.pyi`
- Create: `src/immtrack/metrics/__init__.py`
- Modify: `src/immtrack/__init__.py` (re-export)

- [ ] **Step 1: Update `_core.pyi`**

Append to `src/immtrack/_core.pyi`:

```python
class BoundingBox:
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    rot: float
    class_name: str
    score: float
    track_id: int
    def __init__(
        self,
        x: float = ...,
        y: float = ...,
        z: float = ...,
        l: float = ...,
        w: float = ...,
        h: float = ...,
        rot: float = ...,
        class_name: str = ...,
        score: float = ...,
        track_id: int = ...,
    ) -> None: ...

class TrackedObject:
    id: int
    class_name: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    rot: float
    l: float
    w: float
    h: float
    score: float
    age: int
    hit_count: int
    miss_count: int

class BBoxTracker:
    def __init__(
        self,
        n_init: int = ...,
        max_age: int = ...,
        min_hits: int = ...,
        size_ema_alpha: float = ...,
    ) -> None: ...
    def update(
        self,
        detections: list[BoundingBox],
        dt: float | None = ...,
    ) -> list[TrackedObject]: ...
    def reset(self) -> None: ...
    def track_count(self) -> int: ...

class metrics:
    class MatchMetric:
        CenterDistance: "metrics.MatchMetric"
        Iou3d: "metrics.MatchMetric"

    class AmotaConfig:
        recall_values: list[float]
        metric: "metrics.MatchMetric"
        match_threshold: float
        def __init__(self) -> None: ...

    class AmotaResult:
        overall: float
        per_class: dict[str, float]

    @staticmethod
    def amota(
        gt: list[list[BoundingBox]],
        pred: list[list[BoundingBox]],
        cfg: "metrics.AmotaConfig" = ...,
    ) -> "metrics.AmotaResult": ...
```

- [ ] **Step 2: Create `src/immtrack/metrics/__init__.py`**

```python
"""AMOTA evaluation utilities."""
from immtrack._core.metrics import (
    AmotaConfig,
    AmotaResult,
    MatchMetric,
    amota,
)

__all__ = ["amota", "AmotaConfig", "AmotaResult", "MatchMetric"]
```

- [ ] **Step 3: Update `src/immtrack/__init__.py` to re-export new names**

Read the current file first (it likely re-exports `UkfPosVxyzYawCV`). Add to the re-export list:

```python
from immtrack._core import (
    BBoxTracker,
    BoundingBox,
    TrackedObject,
)
```

(Keep existing re-exports.)

Update `__all__` list to include the new names.

- [ ] **Step 4: Smoke test the import paths**

Run:

```bash
uv run python -c "
from immtrack import BBoxTracker, BoundingBox, TrackedObject
from immtrack.metrics import amota, AmotaConfig, MatchMetric
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/immtrack/_core.pyi src/immtrack/__init__.py src/immtrack/metrics/__init__.py
git commit -m "feat(python): add type stubs and metrics submodule re-exports"
```

---

## Task 12: Python integration tests

**Files:**

- Create: `tests/python/test_bbox_tracker.py`
- Create: `tests/python/test_amota.py`

- [ ] **Step 1: Write `test_bbox_tracker.py`**

```python
"""End-to-end tracker test using the pybind11 binding."""
from __future__ import annotations

import immtrack


def make_box(cls: str, x: float, y: float, z: float = 0.0, yaw: float = 0.0,
             score: float = 0.9) -> immtrack.BoundingBox:
    return immtrack.BoundingBox(
        x=x, y=y, z=z, l=4.0, w=2.0, h=1.5, rot=yaw,
        class_name=cls, score=score,
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
        boxes = [make_box("car", i * 0.5, 0),
                 make_box("pedestrian", i * 0.5, 0)]
        if i == 0:
            tracker.update(boxes)
        else:
            tracker.update(boxes, dt=0.1)
    out = tracker.update([make_box("car", 1.5, 0),
                           make_box("pedestrian", 1.5, 0)], dt=0.1)
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
```

- [ ] **Step 2: Write `test_amota.py`**

```python
"""Smoke test for AMOTA Python binding."""
from __future__ import annotations

from immtrack import BoundingBox
from immtrack.metrics import AmotaConfig, MatchMetric, amota


def gt_box(track_id: int, x: float, y: float, cls: str = "car") -> BoundingBox:
    return BoundingBox(x=x, y=y, z=0.0, l=4.0, w=2.0, h=1.5, rot=0.0,
                       class_name=cls, score=1.0, track_id=track_id)


def pred_box(track_id: int, x: float, y: float, score: float,
              cls: str = "car") -> BoundingBox:
    return BoundingBox(x=x, y=y, z=0.0, l=4.0, w=2.0, h=1.5, rot=0.0,
                       class_name=cls, score=score, track_id=track_id)


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
```

- [ ] **Step 3: Run Python tests, expect pass**

Run: `uv run pytest tests/python -v`
Expected: all tests pass (existing + new).

- [ ] **Step 4: Commit**

```bash
git add tests/python/test_bbox_tracker.py tests/python/test_amota.py
git commit -m "test(python): add tracker and AMOTA integration tests"
```

---

## Task 13: README update

**Files:**

- Modify: `README.md`

- [ ] **Step 1: Add usage section after the existing setup section**

Open `README.md` and append:

````markdown
## Usage

```python
import immtrack

tracker = immtrack.BBoxTracker(n_init=3, max_age=5)

# First frame
boxes_t0 = [
    immtrack.BoundingBox(x=0, y=0, z=0, l=4, w=2, h=1.5, rot=0,
                          class_name="car", score=0.9),
]
tracker.update(boxes_t0)

# Subsequent frames pass dt
boxes_t1 = [
    immtrack.BoundingBox(x=1, y=0, z=0, l=4, w=2, h=1.5, rot=0,
                          class_name="car", score=0.9),
]
confirmed = tracker.update(boxes_t1, dt=0.1)
for t in confirmed:
    print(t.id, t.x, t.y, t.vx, t.vy)
```
````

## Evaluate AMOTA

```python
from immtrack import BoundingBox
from immtrack.metrics import amota, AmotaConfig

# gt[f] and pred[f] are lists of BoundingBox for frame f.
# track_id field must be filled on both sides.
result = amota(gt, pred, AmotaConfig())
print(result.overall)
print(dict(result.per_class))
```

````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document BBoxTracker and AMOTA usage in README"
````

---

## Final verification

After all tasks complete, run the full validation suite:

- [ ] **C++ unit tests**

Run: `ctest --test-dir build-test --output-on-failure`
Expected: all tests pass.

- [ ] **Python tests**

Run: `uv run pytest tests/python -v`
Expected: all tests pass.

- [ ] **Quick integration smoke test**

Run:

```bash
uv run python -c "
import immtrack
from immtrack.metrics import amota
tr = immtrack.BBoxTracker()
b = immtrack.BoundingBox(x=0, y=0, z=0, l=4, w=2, h=1.5, rot=0, class_name='car', score=0.9)
print('first:', tr.update([b]))
print('second:', tr.update([b], dt=0.1))
print('AMOTA empty:', amota([], []).overall)
"
```

Expected: prints empty list, empty list, `0.0`, no errors.

---

## Self-Review

Performed inline. Notes:

- Task 5 introduces `Filter::observation_residual` static helper — consistent
  with existing `Filter::M`, `Filter::MeasVec` exposure pattern.
- Task 7's `BBoxTracker` references `track_matched`/`det_matched` vectors
  sized to `tracks_.size()` and `detections.size()` respectively — matches
  the indexing scheme in steps 3 and 5.
- Task 9's AMOTA cutoff search is O(unique_scores × frames) per recall
  value — acceptable for typical eval sizes (~1000 frames, ~1000 preds).
- ID-switch counting in `count_for_cutoff` only updates `prev_match` from
  current-frame matches; deleted GT IDs naturally drop out, matching
  AB3DMOT semantics.
- All test files use `Catch::Approx` from `catch_approx.hpp`; ensure the
  include is present at the top of each new test file.
