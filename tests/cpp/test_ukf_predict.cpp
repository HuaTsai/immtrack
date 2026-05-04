#include <cmath>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>

#include <immtrack/errors.hpp>
#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <immtrack/ukf.hpp>

#include "eigen_matchers.hpp"

using immtrack::PosVxyzYawCV;
using immtrack::PosYawObs;
using immtrack::UKF;
using immtrack::test::IsApprox;
using immtrack::test::IsPsd;

using Filter = UKF<PosVxyzYawCV, PosYawObs>;

TEST_CASE("Predict on linear motion matches plain motion model", "[predict]") {
    Filter ukf;
    Filter::StateVec x0;
    x0 << 0.0, 0.0, 0.0, 1.0, 2.0, 0.5, 0.0;
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
    Filter ukf;
    Filter::StateVec x0 = Filter::StateVec::Zero();
    x0(6) = M_PI - 0.01;
    Filter::StateMat P0 = Filter::StateMat::Identity() * 0.5;
    P0(6, 6) = 0.5;
    ukf.init(x0, P0);

    ukf.predict(0.01);

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

    Filter::StateMat F = Filter::StateMat::Identity();
    F(0, 3) = 0.1;
    F(1, 4) = 0.1;
    F(2, 5) = 0.1;
    Filter::StateMat expected =
        F * P0 * F.transpose() + Filter::StateMat::Identity() * 0.1;
    REQUIRE_THAT(ukf.covariance(), IsApprox(expected, 1e-7));
}
