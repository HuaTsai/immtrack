#include <catch2/catch_test_macros.hpp>
#include <immtrack/concepts.hpp>
#include <immtrack/state_spaces.hpp>
#include <type_traits>

using immtrack::XYZVxVyVzYawYawRateSpace;

TEST_CASE("XYZVxVyVzYawYawRateSpace satisfies StateSpace", "[state_spaces]") {
  STATIC_REQUIRE(immtrack::StateSpace<XYZVxVyVzYawYawRateSpace>);
  STATIC_REQUIRE(XYZVxVyVzYawYawRateSpace::state_dim == 8);
  STATIC_REQUIRE(XYZVxVyVzYawYawRateSpace::tangent_dim == 8);
}

TEST_CASE("XYZVxVyVzYawYawRateSpace named indices", "[state_spaces]") {
  using S = XYZVxVyVzYawYawRateSpace;
  STATIC_REQUIRE(S::X == 0);
  STATIC_REQUIRE(S::Y == 1);
  STATIC_REQUIRE(S::Z == 2);
  STATIC_REQUIRE(S::VX == 3);
  STATIC_REQUIRE(S::VY == 4);
  STATIC_REQUIRE(S::VZ == 5);
  STATIC_REQUIRE(S::YAW == 6);
  STATIC_REQUIRE(S::YAW_RATE == 7);
}
