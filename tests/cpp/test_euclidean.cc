#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <immtrack/detail/euclidean.hpp>
#include <numbers>

using immtrack::detail::EuclideanWithAngles;

TEST_CASE("EuclideanWithAngles<3>: pure Euclidean (no angle indices)", "[euclidean]") {
  using Space = EuclideanWithAngles<3>;
  STATIC_REQUIRE(Space::state_dim == 3);
  STATIC_REQUIRE(Space::tangent_dim == 3);

  Space::State a;
  a << 1.0, 2.0, 3.0;
  Space::Tangent dx;
  dx << 0.5, -0.5, 1.0;
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
  a(6) = std::numbers::pi - 0.10;
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
  sigmas(0, 0) = 1.0;
  sigmas(0, 1) = 2.0;
  sigmas(0, 2) = 3.0;
  Eigen::Matrix<double, K, 1> w;
  w << 0.5, 0.25, 0.25;
  const auto mean = Space::template weighted_mean<K>(sigmas, w);
  REQUIRE(mean(0) == Catch::Approx(0.5 * 1.0 + 0.25 * 2.0 + 0.25 * 3.0));
}

TEST_CASE("EuclideanWithAngles<8, 6>: weighted_mean handles yaw across pi", "[euclidean]") {
  using Space = EuclideanWithAngles<8, 6>;
  constexpr int K = 2;
  Eigen::Matrix<double, 8, K> sigmas;
  sigmas.setZero();
  sigmas(6, 0) = std::numbers::pi - 0.05;
  sigmas(6, 1) = -std::numbers::pi + 0.05;
  Eigen::Matrix<double, K, 1> w;
  w << 0.5, 0.5;
  const auto mean = Space::template weighted_mean<K>(sigmas, w);
  // Intrinsic mean of (pi - 0.05) and (-pi + 0.05) is +/- pi (wrap point).
  // Result must be wrapped to (-pi, pi]. We accept either +pi or -pi numerically.
  REQUIRE(std::abs(std::abs(mean(6)) - std::numbers::pi) < 1e-10);
}
