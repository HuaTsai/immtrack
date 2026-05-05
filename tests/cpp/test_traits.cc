#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
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
  STATIC_REQUIRE(std::is_same_v<PosVxyzYawCV::Cov, Eigen::Matrix<double, 8, 8>>);
}

TEST_CASE("PosVxyzYawCV::predict propagates position, leaves yaw and yaw_rate", "[traits]") {
  PosVxyzYawCV::State x = PosVxyzYawCV::State::Zero();
  x << 1.0, 2.0, 3.0, 0.5, -0.5, 0.0, 0.7, 0.3;
  const auto next = PosVxyzYawCV::predict(x, 0.5);
  REQUIRE(next(0) == Catch::Approx(1.0 + 0.5 * 0.5));  // x + vx*dt
  REQUIRE(next(1) == Catch::Approx(2.0 - 0.5 * 0.5));  // y + vy*dt
  REQUIRE(next(2) == Catch::Approx(3.0));              // z unchanged (vz=0)
  REQUIRE(next(3) == Catch::Approx(0.5));              // vx unchanged
  REQUIRE(next(4) == Catch::Approx(-0.5));             // vy unchanged
  REQUIRE(next(5) == Catch::Approx(0.0));              // vz unchanged
  REQUIRE(next(6) == Catch::Approx(0.7));              // yaw unchanged (passive in CV)
  REQUIRE(next(7) == Catch::Approx(0.3));              // yaw_rate unchanged (passive in CV)
}

TEST_CASE("PosVxyzYawCV: process_noise zero on yaw_rate row/col", "[traits]") {
  const auto Q = PosVxyzYawCV::process_noise(1.0);
  for (int i = 0; i < 8; ++i) {
    REQUIRE(Q(7, i) == Catch::Approx(0.0));
    REQUIRE(Q(i, 7) == Catch::Approx(0.0));
  }
}

TEST_CASE("PosYawObs trait shape", "[traits]") {
  STATIC_REQUIRE(PosYawObs::M == 4);
  STATIC_REQUIRE(std::is_same_v<PosYawObs::Meas, Eigen::Matrix<double, 4, 1>>);
  STATIC_REQUIRE(std::is_same_v<PosYawObs::Noise, Eigen::Matrix<double, 4, 4>>);
}

TEST_CASE("PosYawObs::h projects position+yaw", "[traits]") {
  PosVxyzYawCV::State x;
  x << 1.0, 2.0, 3.0, 0.5, 0.0, 0.0, 1.57, 0.0;
  const auto z = PosYawObs::h(x);
  REQUIRE(z(0) == 1.0);
  REQUIRE(z(1) == 2.0);
  REQUIRE(z(2) == 3.0);
  REQUIRE(z(3) == 1.57);
}
