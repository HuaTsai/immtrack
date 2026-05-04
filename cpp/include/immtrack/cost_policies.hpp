#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <limits>

#include <immtrack/bbox.hpp>

namespace immtrack {

// Mahalanobis distance using the filter's predicted measurement and
// innovation covariance S. Lower is better. Returns +inf if S is not
// numerically invertible.
//
// Filter requirements:
//   - static constexpr int M (measurement dimension; must be 4 for the
//     [x, y, z, yaw] observation model)
//   - using MeasVec
//   - PredictedMeasurement predict_measurement() const;
//   - static MeasVec observation_residual(const MeasVec&, const MeasVec&)
//     (handles yaw wrapping)
template <class Filter>
struct MahalanobisCost {
    static double cost(const Filter& f, const BoundingBox& d) {
        static_assert(Filter::M == 4,
                      "MahalanobisCost expects 4-D measurement [x, y, z, yaw]");
        const auto pm = f.predict_measurement();

        typename Filter::MeasVec z;
        z << d.x, d.y, d.z, d.rot;

        const typename Filter::MeasVec nu =
            Filter::observation_residual(z, pm.z_pred);

        Eigen::LDLT<Eigen::Matrix<double, Filter::M, Filter::M>> ldlt(pm.S);
        if (ldlt.info() != Eigen::Success) {
            return std::numeric_limits<double>::infinity();
        }
        return nu.dot(ldlt.solve(nu));
    }

    static constexpr double gate_threshold() { return 13.28; }
};

}  // namespace immtrack
