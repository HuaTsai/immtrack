#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Core>
#include <algorithm>
#include <vector>

#include <immtrack/detail/hungarian.hpp>

using Catch::Matchers::WithinAbs;

namespace {

double total_cost(const Eigen::MatrixXd& cost,
                  const std::vector<std::pair<int, int>>& assignment) {
    double total = 0.0;
    for (const auto& [r, c] : assignment) {
        total += cost(r, c);
    }
    return total;
}

}  // namespace

TEST_CASE("Hungarian: empty matrix returns empty assignment", "[hungarian]") {
    Eigen::MatrixXd cost(0, 0);
    const auto a = immtrack::detail::hungarian(cost);
    REQUIRE(a.empty());
}

TEST_CASE("Hungarian: 1x1 returns single pair", "[hungarian]") {
    Eigen::MatrixXd cost(1, 1);
    cost << 3.0;
    const auto a = immtrack::detail::hungarian(cost);
    REQUIRE(a.size() == 1);
    REQUIRE(a[0].first == 0);
    REQUIRE(a[0].second == 0);
}

TEST_CASE("Hungarian: 3x3 known optimal", "[hungarian]") {
    Eigen::MatrixXd cost(3, 3);
    cost << 4.0, 1.0, 3.0,
            2.0, 0.0, 5.0,
            3.0, 2.0, 2.0;
    // Optimal assignment: (0,1), (1,0), (2,2) -> total 1 + 2 + 2 = 5
    const auto a = immtrack::detail::hungarian(cost);
    REQUIRE(a.size() == 3);
    REQUIRE_THAT(total_cost(cost, a), WithinAbs(5.0, 1e-9));
}

TEST_CASE("Hungarian: rectangular 2x3", "[hungarian]") {
    Eigen::MatrixXd cost(2, 3);
    cost << 1.0, 4.0, 5.0,
            7.0, 2.0, 3.0;
    // Optimal: (0,0)=1, (1,1)=2 -> total 3
    const auto a = immtrack::detail::hungarian(cost);
    REQUIRE(a.size() == 2);
    REQUIRE_THAT(total_cost(cost, a), WithinAbs(3.0, 1e-9));
    std::vector<int> rows;
    for (const auto& [r, c] : a) rows.push_back(r);
    std::sort(rows.begin(), rows.end());
    REQUIRE(rows == std::vector<int>{0, 1});
}

TEST_CASE("Hungarian: rectangular 3x2", "[hungarian]") {
    Eigen::MatrixXd cost(3, 2);
    cost << 1.0, 4.0,
            7.0, 2.0,
            5.0, 6.0;
    // Optimal: (0,0)=1, (1,1)=2 -> total 3 (row 2 unmatched)
    const auto a = immtrack::detail::hungarian(cost);
    REQUIRE(a.size() == 2);
    REQUIRE_THAT(total_cost(cost, a), WithinAbs(3.0, 1e-9));
}

TEST_CASE("Hungarian: all cells infeasible still returns valid pairs",
          "[hungarian]") {
    Eigen::MatrixXd cost(2, 2);
    cost << immtrack::detail::kInfeasible, immtrack::detail::kInfeasible,
            immtrack::detail::kInfeasible, immtrack::detail::kInfeasible;
    const auto a = immtrack::detail::hungarian(cost);
    REQUIRE(a.size() == 2);
    // Caller is responsible for filtering >= kInfeasible pairs.
}

TEST_CASE("Hungarian: rectangular with one fully-infeasible row demonstrates "
          "caller filter pattern",
          "[hungarian]") {
    Eigen::MatrixXd cost(2, 3);
    cost << immtrack::detail::kInfeasible, immtrack::detail::kInfeasible,
            immtrack::detail::kInfeasible,
            1.0, 2.0, 3.0;

    const auto raw = immtrack::detail::hungarian(cost);
    REQUIRE(raw.size() == 2);

    // Caller filter: drop pairs with cost >= kInfeasible.
    std::vector<std::pair<int, int>> feasible;
    for (const auto& [r, c] : raw) {
        if (cost(r, c) < immtrack::detail::kInfeasible) {
            feasible.push_back({r, c});
        }
    }

    REQUIRE(feasible.size() == 1);
    REQUIRE(feasible[0].first == 1);
    // Optimal feasible col for row 1 is column 0 (cost 1.0).
    REQUIRE(feasible[0].second == 0);
}
