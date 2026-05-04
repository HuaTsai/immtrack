#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <immtrack/bbox.hpp>
#include <immtrack/detail/iou3d.hpp>

using immtrack::detail::iou3d;

namespace {

immtrack::BoundingBox make(double x,
                           double y,
                           double z,
                           double l,
                           double w,
                           double h,
                           double rot = 0.0) {
    immtrack::BoundingBox b;
    b.x = x;
    b.y = y;
    b.z = z;
    b.l = l;
    b.w = w;
    b.h = h;
    b.rot = rot;
    return b;
}

}  // namespace

TEST_CASE("iou3d: identical boxes give 1.0", "[iou3d]") {
    auto a = make(0, 0, 0, 4, 2, 1.5);
    REQUIRE(iou3d(a, a) == Catch::Approx(1.0).margin(1e-9));
}

TEST_CASE("iou3d: disjoint boxes give 0.0", "[iou3d]") {
    auto a = make(0, 0, 0, 1, 1, 1);
    auto b = make(10, 10, 10, 1, 1, 1);
    REQUIRE(iou3d(a, b) == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("iou3d: half-overlap along x", "[iou3d]") {
    auto a = make(0, 0, 0, 2, 2, 2);  // [-1,1] x [-1,1] x [-1,1]
    auto b = make(1, 0, 0, 2, 2, 2);  // [0,2] x [-1,1] x [-1,1]
    // Intersection: [0,1] x [-1,1] x [-1,1] -> volume 1*2*2 = 4
    // Union: 8 + 8 - 4 = 12
    REQUIRE(iou3d(a, b) == Catch::Approx(4.0 / 12.0).epsilon(1e-9));
}

TEST_CASE("iou3d: rotation changes IoU", "[iou3d]") {
    auto a = make(0, 0, 0, 4, 2, 1.5, 0.0);
    auto b = make(0, 0, 0, 4, 2, 1.5, 1.5707963);  // 90 deg
    const double v = iou3d(a, b);
    REQUIRE(v > 0.0);
    REQUIRE(v < 1.0);
}
