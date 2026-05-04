#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <immtrack/bbox.hpp>
#include <immtrack/metrics/amota.hpp>
#include <string>
#include <vector>

using immtrack::BoundingBox;
using immtrack::metrics::amota;
using immtrack::metrics::AmotaConfig;
using immtrack::metrics::MatchMetric;

namespace {

BoundingBox box(int track_id, double x, double y, double z, double yaw,
                const std::string &cls = "car", double score = 1.0) {
  BoundingBox b;
  b.track_id = track_id;
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

TEST_CASE("amota: empty inputs return 0.0 overall", "[amota]") {
  const auto r = amota({}, {}, {});
  REQUIRE(r.overall == Catch::Approx(0.0));
  REQUIRE(r.per_class.empty());
}

TEST_CASE("amota: perfect predictions yield AMOTA = 1.0", "[amota]") {
  std::vector<std::vector<BoundingBox>> gt = {
      {box(1, 0, 0, 0, 0)},
      {box(1, 1, 0, 0, 0)},
      {box(1, 2, 0, 0, 0)},
  };
  std::vector<std::vector<BoundingBox>> pred = {
      {box(100, 0, 0, 0, 0, "car", 0.9)},
      {box(100, 1, 0, 0, 0, "car", 0.9)},
      {box(100, 2, 0, 0, 0, "car", 0.9)},
  };
  const auto r = amota(gt, pred, {});
  REQUIRE(r.overall == Catch::Approx(1.0).margin(1e-9));
  REQUIRE(r.per_class.at("car") == Catch::Approx(1.0).margin(1e-9));
}

TEST_CASE("amota: pure FP predictions degrade AMOTA", "[amota]") {
  std::vector<std::vector<BoundingBox>> gt = {
      {box(1, 0, 0, 0, 0)},
      {box(1, 1, 0, 0, 0)},
  };
  std::vector<std::vector<BoundingBox>> pred = {
      {box(100, 100, 100, 100, 0, "car", 0.9)},
      {box(100, 200, 200, 200, 0, "car", 0.9)},
  };
  const auto r = amota(gt, pred, {});
  REQUIRE(r.per_class.at("car") < 1.0);
}

TEST_CASE("amota: ID switch is counted", "[amota]") {
  std::vector<std::vector<BoundingBox>> gt = {
      {box(1, 0, 0, 0, 0)},
      {box(1, 1, 0, 0, 0)},
      {box(1, 2, 0, 0, 0)},
  };
  std::vector<std::vector<BoundingBox>> pred = {
      {box(100, 0, 0, 0, 0, "car", 0.9)},
      {box(200, 1, 0, 0, 0, "car", 0.9)},
      {box(200, 2, 0, 0, 0, "car", 0.9)},
  };
  const auto r = amota(gt, pred, {});
  // 1 IDS reduces MOTA at every recall threshold.
  REQUIRE(r.per_class.at("car") < 1.0);
}

TEST_CASE("amota: per-class breakdown for mixed classes", "[amota]") {
  std::vector<std::vector<BoundingBox>> gt = {
      {box(1, 0, 0, 0, 0, "car"), box(2, 5, 5, 0, 0, "pedestrian")},
      {box(1, 1, 0, 0, 0, "car"), box(2, 5, 6, 0, 0, "pedestrian")},
  };
  std::vector<std::vector<BoundingBox>> pred = {
      {box(10, 0, 0, 0, 0, "car", 0.9), box(20, 5, 5, 0, 0, "pedestrian", 0.9)},
      {box(10, 1, 0, 0, 0, "car", 0.9), box(20, 5, 6, 0, 0, "pedestrian", 0.9)},
  };
  const auto r = amota(gt, pred, {});
  REQUIRE(r.per_class.at("car") == Catch::Approx(1.0).margin(1e-9));
  REQUIRE(r.per_class.at("pedestrian") == Catch::Approx(1.0).margin(1e-9));
}

TEST_CASE("amota: Iou3d match metric", "[amota][iou3d]") {
  AmotaConfig cfg;
  cfg.metric = MatchMetric::Iou3d;
  cfg.match_threshold = 0.5;  // need IoU > 0.5

  std::vector<std::vector<BoundingBox>> gt = {
      {box(1, 0, 0, 0, 0)},
  };
  std::vector<std::vector<BoundingBox>> pred = {
      {box(100, 0.1, 0, 0, 0, "car", 0.9)},
  };
  const auto r = amota(gt, pred, cfg);
  REQUIRE(r.per_class.at("car") == Catch::Approx(1.0).margin(1e-9));
}
