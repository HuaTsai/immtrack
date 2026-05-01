#pragma once

#include <cmath>

#include <Eigen/Core>

namespace immtrack {

// State layout shared by all 3D-bbox motion models:
//   [x, y, z, yaw, l, w, h, ...dynamic]
// First 7 dims are the bbox (matches BBox3DObs::h via head<7>()).
// Remaining dims are model-specific velocity / rate / etc.

// Constant velocity in 3D:
//   state = [x, y, z, yaw, l, w, h, vx, vy, vz]   (N = 10)
struct CV3D {
  static constexpr int N = 10;
  using State = Eigen::Matrix<double, N, 1>;
  using Cov = Eigen::Matrix<double, N, N>;

  static State predict(const State &x, double dt) {
    State next = x;
    next(0) += x(7) * dt;
    next(1) += x(8) * dt;
    next(2) += x(9) * dt;
    return next;
  }

  static Cov process_noise(double dt) {
    // TODO: tune properly. Placeholder identity-scaled.
    return Cov::Identity() * dt;
  }
};

// Constant turn-rate / velocity in 3D:
//   state = [x, y, z, yaw, l, w, h, v, yaw_rate, vz]   (N = 10)
struct CTRV3D {
  static constexpr int N = 10;
  using State = Eigen::Matrix<double, N, 1>;
  using Cov = Eigen::Matrix<double, N, N>;

  static State predict(const State &x, double dt) {
    const double yaw = x(3);
    const double v = x(7);
    const double yaw_rate = x(8);
    const double vz = x(9);

    State next = x;
    if (std::abs(yaw_rate) < 1e-6) {
      next(0) += v * std::cos(yaw) * dt;
      next(1) += v * std::sin(yaw) * dt;
    } else {
      const double yaw_next = yaw + yaw_rate * dt;
      next(0) += (v / yaw_rate) * (std::sin(yaw_next) - std::sin(yaw));
      next(1) += (v / yaw_rate) * (-std::cos(yaw_next) + std::cos(yaw));
    }
    next(2) += vz * dt;
    next(3) += yaw_rate * dt;
    return next;
  }

  static Cov process_noise(double dt) {
    return Cov::Identity() * dt;
  }
};

}  // namespace immtrack
