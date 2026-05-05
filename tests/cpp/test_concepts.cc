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
