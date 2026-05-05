#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <immtrack/bbox.hpp>
#include <immtrack/cost_policies.hpp>
#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
#include <immtrack/tracker.hpp>
#include <immtrack/ukf.hpp>
#include <set>
#include <string>
#include <vector>

using Ukf = immtrack::UKF<immtrack::PosVxyzYawCV, immtrack::PosYawObs>;
using Tracker = immtrack::BBoxTracker<Ukf, immtrack::MahalanobisCost>;

namespace {

immtrack::BoundingBox box(const std::string &cls, double x, double y, double z, double yaw,
                          double score = 0.9) {
  immtrack::BoundingBox b;
  b.class_name = cls;
  b.x = x;
  b.y = y;
  b.z = z;
  b.rot = yaw;
  b.l = 4.0;
  b.w = 2.0;
  b.h = 1.5;
  b.score = score;
  return b;
}

}  // namespace

TEST_CASE("BBoxTracker: first frame returns no confirmed tracks", "[tracker]") {
  Tracker tr;
  auto out = tr.update({box("car", 0, 0, 0, 0)});
  // n_init=3 by default, so 1 hit isn't enough to be confirmed.
  REQUIRE(out.empty());
  REQUIRE(tr.track_count() == 1);
}

TEST_CASE("BBoxTracker: single object confirmed after n_init=3 frames", "[tracker]") {
  Tracker tr;
  tr.update({box("car", 0, 0, 0, 0)});
  tr.update({box("car", 1, 0, 0, 0)}, /*dt=*/0.1);
  auto out = tr.update({box("car", 2, 0, 0, 0)}, 0.1);
  REQUIRE(out.size() == 1);
  REQUIRE(out[0].class_name == "car");
  REQUIRE(out[0].id >= 0);
}

TEST_CASE("BBoxTracker: per-class strict (cars and pedestrians don't mix)",
          "[tracker][per-class]") {
  Tracker tr;
  // Spawn one car and one pedestrian at the same location for 3 frames.
  for (int i = 0; i < 3; ++i) {
    const double x = i * 0.5;
    if (i == 0) {
      tr.update({box("car", x, 0, 0, 0), box("pedestrian", x, 0, 0, 0)});
    } else {
      tr.update({box("car", x, 0, 0, 0), box("pedestrian", x, 0, 0, 0)}, 0.1);
    }
  }
  auto out = tr.update({box("car", 1.5, 0, 0, 0), box("pedestrian", 1.5, 0, 0, 0)}, 0.1);
  REQUIRE(out.size() == 2);
  std::set<std::string> classes;
  for (const auto &t : out) classes.insert(t.class_name);
  REQUIRE(classes.count("car") == 1);
  REQUIRE(classes.count("pedestrian") == 1);
}

TEST_CASE("BBoxTracker: occlusion (one missed frame) keeps track alive", "[tracker][occlusion]") {
  Tracker tr;
  tr.update({box("car", 0, 0, 0, 0)});
  tr.update({box("car", 1, 0, 0, 0)}, 0.1);
  auto out = tr.update({box("car", 2, 0, 0, 0)}, 0.1);
  REQUIRE(out.size() == 1);
  const int id = out[0].id;

  // Miss for one frame.
  auto missed = tr.update({}, 0.1);
  REQUIRE(missed.empty());

  // Reappear -- Hungarian reuses the same track.
  auto resumed = tr.update({box("car", 4, 0, 0, 0)}, 0.1);
  REQUIRE(resumed.size() == 1);
  REQUIRE(resumed[0].id == id);
}

TEST_CASE("BBoxTracker: track deleted after max_age misses", "[tracker]") {
  Tracker::Config cfg;
  cfg.max_age = 2;
  Tracker tr(cfg);
  tr.update({box("car", 0, 0, 0, 0)});
  tr.update({box("car", 1, 0, 0, 0)}, 0.1);
  tr.update({box("car", 2, 0, 0, 0)}, 0.1);

  // 3 consecutive misses (max_age=2 => deleted after 3rd miss).
  tr.update({}, 0.1);
  tr.update({}, 0.1);
  tr.update({}, 0.1);

  auto out = tr.update({box("car", 0, 0, 0, 0)}, 0.1);
  // The reappearance creates a new tentative track (not yet confirmed).
  REQUIRE(out.empty());
  REQUIRE(tr.track_count() == 1);  // old deleted, new tentative
}

TEST_CASE("Track init places yaw_rate at idx 7 with finite variance", "[bbox_tracker]") {
  using namespace immtrack;
  using F = UKF<PosVxyzYawCV, PosYawObs>;
  BoundingBox d{};
  d.x = 1.0;
  d.y = 2.0;
  d.z = 3.0;
  d.rot = 0.4;
  d.l = 4.0;
  d.w = 1.8;
  d.h = 1.6;
  d.score = 0.9;
  d.class_name = "car";
  Track<F> t(/*id=*/0, d);
  const auto &x = t.filter().state();
  REQUIRE(x(F::StateSpace::YAW_RATE) == Catch::Approx(0.0));
  const auto &P = t.filter().covariance();
  REQUIRE(P(F::StateSpace::YAW_RATE, F::StateSpace::YAW_RATE) > 0.0);
}

TEST_CASE("BBoxTracker: reset clears all tracks and resets ID counter", "[tracker][reset]") {
  Tracker tr;
  tr.update({box("car", 0, 0, 0, 0)});
  REQUIRE(tr.track_count() == 1);
  tr.reset();
  REQUIRE(tr.track_count() == 0);
}
