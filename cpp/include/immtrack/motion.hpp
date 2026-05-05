#pragma once

#include <Eigen/Core>
#include <immtrack/state_spaces.hpp>

namespace immtrack {

// Constant-velocity model in 3D Cartesian space + passive yaw + passive yaw_rate.
// state = [x, y, z, vx, vy, vz, yaw, yaw_rate]
//
// CV leaves yaw and yaw_rate unchanged across predict; their covariance grows
// only through process_noise on (x,y,z) (white-noise acceleration in xy/z).
// In Phase 2 a CTRV mode will share the same StateSpace and write yaw/yaw_rate.
struct PosVxyzYawCV {
  using StateSpace = XYZVxVyVzYawYawRateSpace;
  using State = StateSpace::State;
  using Cov = StateSpace::Cov;
  static constexpr int N = StateSpace::state_dim;

  static State predict(const State &x, double dt) noexcept {
    State next = x;
    next(StateSpace::X) += x(StateSpace::VX) * dt;
    next(StateSpace::Y) += x(StateSpace::VY) * dt;
    next(StateSpace::Z) += x(StateSpace::VZ) * dt;
    return next;
  }

  // White-noise acceleration model on (x,y,z); zero on yaw and yaw_rate so
  // those dimensions stay frozen in CV mode.
  static Cov process_noise(double dt) noexcept {
    Cov Q = Cov::Zero();
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    constexpr double sigma_a2 = 1.0;  // matches existing tuning (Q ~ I*dt for v).
    // Position-position blocks.
    Q(StateSpace::X, StateSpace::X) = (dt3 / 3.0) * sigma_a2;
    Q(StateSpace::Y, StateSpace::Y) = (dt3 / 3.0) * sigma_a2;
    Q(StateSpace::Z, StateSpace::Z) = (dt3 / 3.0) * sigma_a2;
    // Position-velocity cross blocks.
    Q(StateSpace::X, StateSpace::VX) = Q(StateSpace::VX, StateSpace::X) = (dt2 / 2.0) * sigma_a2;
    Q(StateSpace::Y, StateSpace::VY) = Q(StateSpace::VY, StateSpace::Y) = (dt2 / 2.0) * sigma_a2;
    Q(StateSpace::Z, StateSpace::VZ) = Q(StateSpace::VZ, StateSpace::Z) = (dt2 / 2.0) * sigma_a2;
    // Velocity-velocity blocks.
    Q(StateSpace::VX, StateSpace::VX) = dt * sigma_a2;
    Q(StateSpace::VY, StateSpace::VY) = dt * sigma_a2;
    Q(StateSpace::VZ, StateSpace::VZ) = dt * sigma_a2;
    // yaw / yaw_rate rows and columns left at zero.
    return Q;
  }

  static Cov F_matrix(double dt) noexcept {
    Cov F = Cov::Identity();
    F(StateSpace::X, StateSpace::VX) = dt;
    F(StateSpace::Y, StateSpace::VY) = dt;
    F(StateSpace::Z, StateSpace::VZ) = dt;
    return F;
  }

  // ===== UKF compatibility shims (delegate to StateSpace) =====
  // The existing UKF reads `Motion::weighted_mean` and `Motion::residual`.
  // Task 6 may switch UKF to call StateSpace directly; until then these
  // shims keep the existing call sites working unchanged.
  template <int K>
  static State weighted_mean(const Eigen::Matrix<double, N, K> &sigmas,
                             const Eigen::Matrix<double, K, 1> &weights) noexcept {
    return StateSpace::template weighted_mean<K>(sigmas, weights);
  }

  static State residual(const State &a, const State &b) noexcept {
    return StateSpace::boxminus(a, b);
  }
};

}  // namespace immtrack
