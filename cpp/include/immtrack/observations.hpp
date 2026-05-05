#pragma once

#include <Eigen/Core>
#include <immtrack/concepts.hpp>
#include <immtrack/detail/euclidean.hpp>
#include <immtrack/state_spaces.hpp>

namespace immtrack {

// Measurement space: [x, y, z, yaw]. Yaw at idx 3 wraps to (-pi, pi].
using PosYawMeasSpace = detail::EuclideanWithAngles<4, /*yaw=*/3>;

// Observation: project (x, y, z, yaw) out of the 8D ground-vehicle state.
struct PosYawObs {
  using StateSpace = XYZVxVyVzYawYawRateSpace;
  using MeasSpace = PosYawMeasSpace;
  using Meas = MeasSpace::State;
  using Noise = MeasSpace::Cov;
  using HMat = ObsJacMat<PosYawObs>;  // 4 x 8
  static constexpr int M = MeasSpace::state_dim;

  static Meas h(const StateSpace::State &x) noexcept {
    Meas z;
    z(0) = x(StateSpace::X);
    z(1) = x(StateSpace::Y);
    z(2) = x(StateSpace::Z);
    z(3) = x(StateSpace::YAW);
    return z;
  }

  static Noise measurement_noise() noexcept { return Noise::Identity(); }

  static HMat H_matrix() noexcept {
    HMat H = HMat::Zero();
    H(0, StateSpace::X) = 1.0;
    H(1, StateSpace::Y) = 1.0;
    H(2, StateSpace::Z) = 1.0;
    H(3, StateSpace::YAW) = 1.0;
    return H;
  }

  // ===== UKF compatibility shims (delegate to MeasSpace) =====
  template <int K>
  static Meas weighted_mean(const Eigen::Matrix<double, M, K> &sigmas,
                            const Eigen::Matrix<double, K, 1> &weights) noexcept {
    return MeasSpace::template weighted_mean<K>(sigmas, weights);
  }

  static Meas residual(const Meas &a, const Meas &b) noexcept { return MeasSpace::boxminus(a, b); }
};

}  // namespace immtrack
