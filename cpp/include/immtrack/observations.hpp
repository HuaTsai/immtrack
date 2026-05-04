#pragma once

#include <cmath>

#include <Eigen/Core>

#include <immtrack/detail/angle.hpp>

namespace immtrack {

// 3D position + yaw observation: measurement = [x, y, z, theta].
struct PosYawObs {
    static constexpr int M = 4;
    using Meas = Eigen::Matrix<double, M, 1>;
    using Noise = Eigen::Matrix<double, M, M>;

    template <class State>
    static Meas h(const State& x) {
        Meas z;
        z(0) = x(0);
        z(1) = x(1);
        z(2) = x(2);
        z(3) = x(6);
        return z;
    }

    static Noise measurement_noise() {
        return Noise::Identity();
    }

    template <int K>
    static Meas weighted_mean(
        const Eigen::Matrix<double, M, K>& sigmas,
        const Eigen::Matrix<double, K, 1>& weights) {
        Meas mean = sigmas.col(0);
        mean.template head<3>().setZero();
        for (int i = 0; i < K; ++i) {
            mean.template head<3>() +=
                weights(i) *
                (sigmas.col(i).template head<3>() -
                 sigmas.col(0).template head<3>());
        }
        mean.template head<3>() += sigmas.col(0).template head<3>();

        const double theta0 = sigmas(3, 0);
        double theta_delta = 0.0;
        for (int i = 0; i < K; ++i) {
            theta_delta +=
                weights(i) * detail::wrap_angle(sigmas(3, i) - theta0);
        }
        mean(3) = detail::wrap_angle(theta0 + theta_delta);
        return mean;
    }

    static Meas residual(const Meas& a, const Meas& b) {
        Meas r = a - b;
        r(3) = detail::wrap_angle(r(3));
        return r;
    }
};

}  // namespace immtrack
