#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <immtrack/detail/sigma_points.hpp>
#include <immtrack/errors.hpp>

namespace immtrack {

// Unscented Kalman Filter parameterised by Motion and Obs traits.
//
// Trait contract (compile-time):
//
//   class Motion {
//       static constexpr int N;
//       using State = Eigen::Matrix<double, N, 1>;
//       using Cov   = Eigen::Matrix<double, N, N>;
//
//       static State predict(const State &x, double dt);
//       static Cov   process_noise(double dt);
//
//       template <int K>
//       static State weighted_mean(
//           const Eigen::Matrix<double, N, K> &sigmas,
//           const Eigen::Matrix<double, K, 1> &weights);
//
//       static State residual(const State &a, const State &b);
//   };
//
//   class Obs {
//       static constexpr int M;
//       using Meas  = Eigen::Matrix<double, M, 1>;
//       using Noise = Eigen::Matrix<double, M, M>;
//
//       template <class State> static Meas h(const State &x);
//       static Noise measurement_noise();
//
//       template <int K>
//       static Meas weighted_mean(
//           const Eigen::Matrix<double, M, K> &sigmas,
//           const Eigen::Matrix<double, K, 1> &weights);
//
//       static Meas residual(const Meas &a, const Meas &b);
//   };
//
// Default constructor: alpha=1e-3, beta=2, kappa=0 (Merwe scaled).
// Default state: mu = 0, Sigma = I.
// update(z) returns NIS (normalized innovation squared).
template <class Motion, class Obs>
class UKF {
 public:
  static constexpr int N = Motion::N;
  static constexpr int M = Obs::M;
  static constexpr int K = 2 * N + 1;
  using StateVec = Eigen::Matrix<double, N, 1>;
  using StateMat = Eigen::Matrix<double, N, N>;
  using MeasVec = Eigen::Matrix<double, M, 1>;

  struct PredictedMeasurement {
    MeasVec z_pred;
    Eigen::Matrix<double, M, M> S;
  };

  UKF() : UKF(1e-3, 2.0, 0.0) {}

  UKF(double alpha, double beta, double kappa)
      : x_(StateVec::Zero()),
        P_(StateMat::Identity()),
        weights_(detail::UnscentedWeights<N>::make(alpha, beta, kappa)) {}

  void init(const StateVec &x, const StateMat &P) {
    x_ = x;
    P_ = P;
  }

  void predict(double dt) {
    if (dt < 0.0) {
      throw InvalidArgument("UKF::predict: dt must be non-negative");
    }

    const auto sigmas = detail::generate_sigma_points<N>(x_, P_, weights_.lambda);

    Eigen::Matrix<double, N, K> propagated;
    for (int i = 0; i < K; ++i) {
      propagated.col(i) = Motion::predict(sigmas.col(i), dt);
    }

    const StateVec x_new = Motion::template weighted_mean<K>(propagated, weights_.mean_weights);

    StateMat P_new = StateMat::Zero();
    for (int i = 0; i < K; ++i) {
      const StateVec d = Motion::residual(propagated.col(i), x_new);
      P_new.noalias() += weights_.cov_weights(i) * d * d.transpose();
    }
    P_new += Motion::process_noise(dt);

    x_ = x_new;
    P_ = 0.5 * (P_new + P_new.transpose());
  }

  double update(const MeasVec &z) {
    const auto c = compute_innovation_();

    using MeasMat = Eigen::Matrix<double, M, M>;
    using CrossMat = Eigen::Matrix<double, N, M>;

    CrossMat T = CrossMat::Zero();
    for (int i = 0; i < K; ++i) {
      const MeasVec dz = Obs::residual(c.z_sigmas.col(i), c.z_pred);
      const StateVec dx = Motion::residual(c.sigmas.col(i), x_);
      T.noalias() += weights_.cov_weights(i) * dx * dz.transpose();
    }

    Eigen::LDLT<MeasMat> S_ldlt(c.S);
    if (S_ldlt.info() != Eigen::Success) {
      throw NumericalError("UKF::update: innovation covariance not invertible");
    }

    const CrossMat K_gain = S_ldlt.solve(T.transpose()).transpose();
    const MeasVec innovation = Obs::residual(z, c.z_pred);

    x_.noalias() += K_gain * innovation;
    const StateMat P_post = P_ - K_gain * c.S * K_gain.transpose();
    P_ = 0.5 * (P_post + P_post.transpose());

    return innovation.dot(S_ldlt.solve(innovation));
  }

  // Compute predicted measurement and innovation covariance without
  // mutating filter state. The returned `S` includes measurement noise
  // and is symmetrised. Used for gating / cost computation in tracking.
  PredictedMeasurement predict_measurement() const {
    const auto c = compute_innovation_();
    using MeasMat = Eigen::Matrix<double, M, M>;
    const MeasMat S_sym = 0.5 * (c.S + c.S.transpose());
    return PredictedMeasurement{c.z_pred, S_sym};
  }

  static MeasVec observation_residual(const MeasVec &a, const MeasVec &b) {
    return Obs::residual(a, b);
  }

  const StateVec &state() const noexcept { return x_; }
  const StateMat &covariance() const noexcept { return P_; }

 private:
  StateVec x_;
  StateMat P_;
  detail::UnscentedWeights<N> weights_;

  struct InnovationCache {
    Eigen::Matrix<double, N, K> sigmas;
    Eigen::Matrix<double, M, K> z_sigmas;
    MeasVec z_pred;
    Eigen::Matrix<double, M, M> S;  // includes R, NOT yet symmetrised
  };

  InnovationCache compute_innovation_() const {
    InnovationCache c;
    c.sigmas = detail::generate_sigma_points<N>(x_, P_, weights_.lambda);
    for (int i = 0; i < K; ++i) {
      c.z_sigmas.col(i) = Obs::h(c.sigmas.col(i));
    }
    c.z_pred = Obs::template weighted_mean<K>(c.z_sigmas, weights_.mean_weights);

    c.S.setZero();
    for (int i = 0; i < K; ++i) {
      const MeasVec dz = Obs::residual(c.z_sigmas.col(i), c.z_pred);
      c.S.noalias() += weights_.cov_weights(i) * dz * dz.transpose();
    }
    c.S += Obs::measurement_noise();
    return c;
  }
};

}  // namespace immtrack
