#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace immtrack::detail {

// Rectangular Hungarian (Jonker-Volgenant style) minimum-cost assignment.
// Returns vector of (row, col) pairs of size = min(rows, cols).
//
// Complexity: O(n^3) where n = max(rows, cols) after square padding.
// Padding fill value is 0.0 (caller should ensure real costs are finite
// and non-negative for the unpadded portion; INFEASIBLE cells should use
// a large constant and be filtered post-call).
inline std::vector<std::pair<int, int>> hungarian(
    const Eigen::MatrixXd& cost) {
    const int rows = static_cast<int>(cost.rows());
    const int cols = static_cast<int>(cost.cols());
    if (rows == 0 || cols == 0) {
        return {};
    }

    const int n = std::max(rows, cols);
    // Pad to n x n with 0.0 in slack cells (1-indexed: a[i][j] for i,j in [1,n]).
    std::vector<std::vector<double>> a(n + 1,
                                        std::vector<double>(n + 1, 0.0));
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (i <= rows && j <= cols) {
                a[i][j] = cost(i - 1, j - 1);
            }
        }
    }

    constexpr double INF = std::numeric_limits<double>::infinity();
    std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, INF);
        std::vector<char> used(n + 1, false);
        do {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = INF;
            int j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    const double cur = a[i0][j] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> ans(n + 1, -1);
    for (int j = 1; j <= n; ++j) {
        if (p[j] != 0) {
            ans[p[j]] = j;
        }
    }

    std::vector<std::pair<int, int>> result;
    result.reserve(std::min(rows, cols));
    for (int i = 1; i <= rows; ++i) {
        const int j = ans[i];
        if (j >= 1 && j <= cols) {
            result.emplace_back(i - 1, j - 1);
        }
    }
    return result;
}

}  // namespace immtrack::detail
