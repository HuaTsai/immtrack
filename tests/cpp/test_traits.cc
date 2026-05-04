#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <type_traits>

using immtrack::PosVxyzYawCV;
using immtrack::PosYawObs;

TEST_CASE("PosVxyzYawCV trait shape", "[traits]") {
  STATIC_REQUIRE(PosVxyzYawCV::N == 7);
  STATIC_REQUIRE(std::is_same_v<PosVxyzYawCV::State, Eigen::Matrix<double, 7, 1>>);
  STATIC_REQUIRE(std::is_same_v<PosVxyzYawCV::Cov, Eigen::Matrix<double, 7, 7>>);
}

TEST_CASE("PosYawObs trait shape", "[traits]") {
  STATIC_REQUIRE(PosYawObs::M == 4);
  STATIC_REQUIRE(std::is_same_v<PosYawObs::Meas, Eigen::Matrix<double, 4, 1>>);
  STATIC_REQUIRE(std::is_same_v<PosYawObs::Noise, Eigen::Matrix<double, 4, 4>>);
}

TEST_CASE("PosVxyzYawCV::predict propagates position", "[traits]") {
  PosVxyzYawCV::State x = PosVxyzYawCV::State::Zero();
  x(3) = 2.0;
  const auto next = PosVxyzYawCV::predict(x, 0.5);
  REQUIRE(next(0) == 1.0);
  REQUIRE(next(1) == 0.0);
  REQUIRE(next(3) == 2.0);
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
