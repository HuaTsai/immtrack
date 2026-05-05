#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <immtrack/concepts.hpp>

namespace {

// Minimal stub satisfying StateSpace.
struct StubSpace {
  using Scalar = double;
  using State = Eigen::Matrix<double, 3, 1>;
  using Tangent = Eigen::Matrix<double, 3, 1>;
  using Cov = Eigen::Matrix<double, 3, 3>;
  static constexpr int state_dim = 3;
  static constexpr int tangent_dim = 3;
  static State boxplus(const State &x, const Tangent &dx) noexcept { return x + dx; }
  static Tangent boxminus(const State &a, const State &b) noexcept { return a - b; }
};

// Minimal motion (linear) over StubSpace.
struct StubMotion {
  using StateSpace = StubSpace;
  static StateSpace::State predict(const StateSpace::State &x, double) noexcept { return x; }
  static StateSpace::Cov process_noise(double) noexcept { return StateSpace::Cov::Identity(); }
  static StateSpace::Cov F_jacobian(const StateSpace::State &, double) noexcept {
    return StateSpace::Cov::Identity();
  }
  static StateSpace::Cov F_matrix(double) noexcept { return StateSpace::Cov::Identity(); }
};

// 2D measurement space.
struct StubMeasSpace {
  using Scalar = double;
  using State = Eigen::Matrix<double, 2, 1>;
  using Tangent = Eigen::Matrix<double, 2, 1>;
  using Cov = Eigen::Matrix<double, 2, 2>;
  static constexpr int state_dim = 2;
  static constexpr int tangent_dim = 2;
  static State boxplus(const State &x, const Tangent &dx) noexcept { return x + dx; }
  static Tangent boxminus(const State &a, const State &b) noexcept { return a - b; }
};

// Motion exposing only F_matrix (no F_jacobian). Used to verify that
// LinearMotion does NOT imply LinearizableMotion in the sibling design.
struct StubLinearOnlyMotion {
  using StateSpace = StubSpace;
  static StateSpace::State predict(const StateSpace::State &x, double) noexcept { return x; }
  static StateSpace::Cov process_noise(double) noexcept { return StateSpace::Cov::Identity(); }
  static StateSpace::Cov F_matrix(double) noexcept { return StateSpace::Cov::Identity(); }
};

// Motion exposing only F_jacobian (no F_matrix). Should be LinearizableMotion only.
struct StubLinearizableOnlyMotion {
  using StateSpace = StubSpace;
  static StateSpace::State predict(const StateSpace::State &x, double) noexcept { return x; }
  static StateSpace::Cov process_noise(double) noexcept { return StateSpace::Cov::Identity(); }
  static StateSpace::Cov F_jacobian(const StateSpace::State &, double) noexcept {
    return StateSpace::Cov::Identity();
  }
};

// Minimal obs (linear) over StubSpace -> StubMeasSpace.
struct StubObs {
  using StateSpace = StubSpace;
  using MeasSpace = StubMeasSpace;
  using HMat = Eigen::Matrix<double, 2, 3>;
  static MeasSpace::State h(const StateSpace::State &x) noexcept { return x.head<2>(); }
  static MeasSpace::Cov measurement_noise() noexcept { return MeasSpace::Cov::Identity(); }
  static HMat H_jacobian(const StateSpace::State &) noexcept { return H_matrix(); }
  static HMat H_matrix() noexcept {
    HMat H = HMat::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    return H;
  }
};

struct StubLinearOnlyObs {
  using StateSpace = StubSpace;
  using MeasSpace = StubMeasSpace;
  using HMat = Eigen::Matrix<double, 2, 3>;
  static MeasSpace::State h(const StateSpace::State &x) noexcept { return x.head<2>(); }
  static MeasSpace::Cov measurement_noise() noexcept { return MeasSpace::Cov::Identity(); }
  static HMat H_matrix() noexcept {
    HMat H = HMat::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    return H;
  }
};

struct StubLinearizableOnlyObs {
  using StateSpace = StubSpace;
  using MeasSpace = StubMeasSpace;
  using HMat = Eigen::Matrix<double, 2, 3>;
  static MeasSpace::State h(const StateSpace::State &x) noexcept { return x.head<2>(); }
  static MeasSpace::Cov measurement_noise() noexcept { return MeasSpace::Cov::Identity(); }
  static HMat H_jacobian(const StateSpace::State &) noexcept {
    HMat H = HMat::Zero();
    H(0, 0) = 1;
    H(1, 1) = 1;
    return H;
  }
};

}  // namespace

TEST_CASE("Manifold concept", "[concepts]") {
  STATIC_REQUIRE(immtrack::Manifold<StubSpace>);
  STATIC_REQUIRE(immtrack::Manifold<StubMeasSpace>);
}

TEST_CASE("Motion concepts", "[concepts]") {
  // Stub providing both F_matrix and F_jacobian satisfies all three.
  STATIC_REQUIRE(immtrack::Motion<StubMotion>);
  STATIC_REQUIRE(immtrack::LinearizableMotion<StubMotion>);
  STATIC_REQUIRE(immtrack::LinearMotion<StubMotion>);

  // LinearMotion and LinearizableMotion are siblings of Motion, not a chain:
  // each can be satisfied independently.
  STATIC_REQUIRE(immtrack::LinearMotion<StubLinearOnlyMotion>);
  STATIC_REQUIRE_FALSE(immtrack::LinearizableMotion<StubLinearOnlyMotion>);

  STATIC_REQUIRE(immtrack::LinearizableMotion<StubLinearizableOnlyMotion>);
  STATIC_REQUIRE_FALSE(immtrack::LinearMotion<StubLinearizableOnlyMotion>);
}

TEST_CASE("Observation concepts", "[concepts]") {
  STATIC_REQUIRE(immtrack::Observation<StubObs>);
  STATIC_REQUIRE(immtrack::LinearizableObservation<StubObs>);
  STATIC_REQUIRE(immtrack::LinearObservation<StubObs>);

  STATIC_REQUIRE(immtrack::LinearObservation<StubLinearOnlyObs>);
  STATIC_REQUIRE_FALSE(immtrack::LinearizableObservation<StubLinearOnlyObs>);

  STATIC_REQUIRE(immtrack::LinearizableObservation<StubLinearizableOnlyObs>);
  STATIC_REQUIRE_FALSE(immtrack::LinearObservation<StubLinearizableOnlyObs>);
}
