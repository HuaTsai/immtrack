#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
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

TEST_CASE(
    "Update with predicted measurement leaves state approximately "
    "unchanged and shrinks cov",
    "[update]") {
  Filter ukf;
  Filter::StateVec x0;
  x0 << 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.5, 0.0;
  Filter::StateMat P0 = Filter::StateMat::Identity() * 0.5;
  ukf.init(x0, P0);

  Filter::MeasVec z;
  z << 1.0, 2.0, 3.0, 0.5;

  const double trace_before = ukf.covariance().trace();
  const double nis = ukf.update(z);
  const double trace_after = ukf.covariance().trace();

  REQUIRE_THAT(ukf.state(), IsApprox(x0, 1e-9));
  REQUIRE(trace_after < trace_before);
  REQUIRE(nis >= 0.0);
  REQUIRE_THAT(ukf.covariance(), IsPsd());
}

TEST_CASE("Update theta residual wraps across +/- pi boundary", "[update]") {
  Filter ukf;
  Filter::StateVec x0 = Filter::StateVec::Zero();
  x0(6) = M_PI - 0.05;
  Filter::StateMat P0 = Filter::StateMat::Identity() * 0.1;
  ukf.init(x0, P0);

  Filter::MeasVec z;
  z << 0.0, 0.0, 0.0, -M_PI + 0.05;

  const double theta_before = ukf.state()(6);
  ukf.update(z);
  const double theta_after = ukf.state()(6);

  const double delta = theta_after - theta_before;
  REQUIRE(std::abs(delta) < 0.2);
}

TEST_CASE("Update converges to ground truth under repeated measurements", "[update]") {
  Filter ukf(1e-3, 2.0, 0.0);
  Filter::StateVec x0 = Filter::StateVec::Zero();
  Filter::StateMat P0 = Filter::StateMat::Identity() * 5.0;
  ukf.init(x0, P0);

  const Filter::MeasVec z_true = (Filter::MeasVec() << 2.0, -1.0, 0.5, 0.3).finished();

  for (int i = 0; i < 50; ++i) {
    ukf.predict(0.0);
    ukf.update(z_true);
  }

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
