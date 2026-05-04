#pragma once

#include <Eigen/Core>
#include <cmath>
#include <immtrack/detail/angle.hpp>

namespace immtrack {

// Constant-velocity model in 3D Cartesian space with passive yaw.
// state = [x, y, z, vx, vy, vz, theta]
struct PosVxyzYawCV {
  static constexpr int N = 7;
  using State = Eigen::Matrix<double, N, 1>;
  using Cov = Eigen::Matrix<double, N, N>;

  static State predict(const State &x, double dt) {
    State next = x;
    next(0) += x(3) * dt;
    next(1) += x(4) * dt;
    next(2) += x(5) * dt;
    return next;
  }

  static Cov process_noise(double dt) { return Cov::Identity() * dt; }

  template <int K>
  static State weighted_mean(const Eigen::Matrix<double, N, K> &sigmas,
                             const Eigen::Matrix<double, K, 1> &weights) {
    State mean = sigmas.col(0);
    mean.template head<6>().setZero();
    for (int i = 0; i < K; ++i) {
      mean.template head<6>() +=
          weights(i) * (sigmas.col(i).template head<6>() - sigmas.col(0).template head<6>());
    }
    mean.template head<6>() += sigmas.col(0).template head<6>();

    const double theta0 = sigmas(6, 0);
    double theta_delta = 0.0;
    for (int i = 0; i < K; ++i) {
      theta_delta += weights(i) * detail::wrap_angle(sigmas(6, i) - theta0);
    }
    mean(6) = detail::wrap_angle(theta0 + theta_delta);
    return mean;
  }

  static State residual(const State &a, const State &b) {
    State r = a - b;
    r(6) = detail::wrap_angle(r(6));
    return r;
  }
};

}  // namespace immtrack
