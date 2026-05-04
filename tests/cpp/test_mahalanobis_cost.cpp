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
