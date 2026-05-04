#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <immtrack/detail/angle.hpp>
#include <limits>

using Catch::Matchers::WithinAbs;
using immtrack::detail::wrap_angle;

TEST_CASE("wrap_angle: in-range values unchanged", "[angle]") {
  REQUIRE_THAT(wrap_angle(0.0), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(wrap_angle(1.0), WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(wrap_angle(-1.0), WithinAbs(-1.0, 1e-12));
}

TEST_CASE("wrap_angle: just past +pi wraps to negative", "[angle]") {
  const double a = M_PI + 0.1;
  REQUIRE_THAT(wrap_angle(a), WithinAbs(-M_PI + 0.1, 1e-12));
}

TEST_CASE("wrap_angle: just past -pi wraps to positive", "[angle]") {
  const double a = -M_PI - 0.1;
  REQUIRE_THAT(wrap_angle(a), WithinAbs(M_PI - 0.1, 1e-12));
}

TEST_CASE("wrap_angle: multi-revolution input", "[angle]") {
  REQUIRE_THAT(wrap_angle(4 * M_PI + 0.5), WithinAbs(0.5, 1e-12));
  REQUIRE_THAT(wrap_angle(-4 * M_PI - 0.5), WithinAbs(-0.5, 1e-12));
}

TEST_CASE("wrap_angle: NaN propagates to NaN", "[angle]") {
  REQUIRE(std::isnan(wrap_angle(std::numeric_limits<double>::quiet_NaN())));
}
