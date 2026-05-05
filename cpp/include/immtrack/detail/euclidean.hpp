#pragma once

#include <Eigen/Core>
#include <immtrack/detail/angle.hpp>

namespace immtrack::detail {

// Euclidean state space with angle-wrapped indices.
//
// All non-angle dimensions are plain Euclidean. Each index listed in
// `AngleIdx` lives on S^1 (theta in (-pi, pi]) and uses `wrap_angle`
// for boxplus / boxminus / weighted_mean.
//
// Used as the implementation backbone of named StateSpaces in
// state_spaces.hpp (e.g. XYZVxVyVzYawYawRateSpace = EuclideanWithAngles<8, 6>).
template <int Dim, int... AngleIdx>
struct EuclideanWithAngles {
  using Scalar = double;
  using State = Eigen::Matrix<double, Dim, 1>;
  using Tangent = Eigen::Matrix<double, Dim, 1>;
  using Cov = Eigen::Matrix<double, Dim, Dim>;
  static constexpr int state_dim = Dim;
  static constexpr int tangent_dim = Dim;

  static State boxplus(const State &x, const Tangent &dx) noexcept {
    State r = x + dx;
    ((r(AngleIdx) = wrap_angle(r(AngleIdx))), ...);
    return r;
  }

  static Tangent boxminus(const State &a, const State &b) noexcept {
    Tangent r = a - b;
    ((r(AngleIdx) = wrap_angle(r(AngleIdx))), ...);
    return r;
  }

  // Weighted mean on the manifold:
  //   - non-angle dimensions: ordinary weighted Euclidean mean.
  //   - each angle dimension: intrinsic mean using wrap_angle deltas
  //     anchored at sigmas.col(0)'s value (numerically stable for
  //     small spreads typical of UKF sigma points).
  template <int K>
  static State weighted_mean(const Eigen::Matrix<double, Dim, K> &sigmas,
                             const Eigen::Matrix<double, K, 1> &weights) noexcept {
    State mean = State::Zero();
    // Plain weighted sum over all dims.
    for (int i = 0; i < K; ++i) {
      mean.noalias() += weights(i) * sigmas.col(i);
    }
    // Recompute angle dims as intrinsic mean (overwrites the linear
    // sum above for those indices).
    ((mean(AngleIdx) = intrinsic_angle_mean_<K>(sigmas, weights, AngleIdx)), ...);
    return mean;
  }

 private:
  template <int K>
  static double intrinsic_angle_mean_(const Eigen::Matrix<double, Dim, K> &sigmas,
                                      const Eigen::Matrix<double, K, 1> &weights,
                                      int idx) noexcept {
    const double anchor = sigmas(idx, 0);
    double delta = 0.0;
    for (int i = 0; i < K; ++i) {
      delta += weights(i) * wrap_angle(sigmas(idx, i) - anchor);
    }
    return wrap_angle(anchor + delta);
  }
};

}  // namespace immtrack::detail
