#include <catch2/catch_approx.hpp>
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
