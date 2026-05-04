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

    Eigen::LDLT<Eigen::Matrix<double, 4, 4>> ldlt(pm.S);
    REQUIRE(ldlt.info() == Eigen::Success);
    REQUIRE(pm.S.isApprox(pm.S.transpose(), 1e-12));

    REQUIRE(ukf.state().isApprox(state_before, 1e-15));
    REQUIRE(ukf.covariance().isApprox(cov_before, 1e-15));
}

TEST_CASE("predict_measurement is idempotent (no hidden state)",
          "[ukf][predict_measurement]") {
    Ukf ukf;
    Ukf::StateVec x;
    x << 0.5, -1.0, 2.0, 0.3, -0.1, 0.05, 0.7;
    Ukf::StateMat P = Ukf::StateMat::Identity() * 0.3;
    ukf.init(x, P);

    const auto pm1 = ukf.predict_measurement();
    const auto pm2 = ukf.predict_measurement();

    REQUIRE(pm1.z_pred.isApprox(pm2.z_pred, 1e-15));
    REQUIRE(pm1.S.isApprox(pm2.S, 1e-15));
}
