#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <immtrack/concepts.hpp>
#include <immtrack/detail/sigma_points.hpp>
#include <immtrack/errors.hpp>

namespace immtrack {

// Unscented Kalman Filter constrained on the MotionModel and ObservationModel
// concepts defined in immtrack/concepts.hpp.
//
// Concept contract (compile-time):
//   Motion must satisfy MotionModel<Motion>:
//     - exposes Motion::StateSpace (satisfies StateSpace concept)
//     - static State predict(const State&, Scalar dt)
//     - static Cov   process_noise(Scalar dt)
//     - static State weighted_mean<K>(sigmas, weights)
//     - static State residual(a, b)
//
//   Obs must satisfy ObservationModel<Obs>:
//     - exposes Obs::StateSpace (must equal Motion::StateSpace)
//     - exposes Obs::MeasSpace  (satisfies StateSpace concept)
//     - static MeasSpace::State h(const StateSpace::State&)
//     - static MeasSpace::Cov   measurement_noise()
//     - static MeasSpace::State weighted_mean<K>(sigmas, weights)
//     - static MeasSpace::State residual(a, b)
//
// Default constructor: alpha=1e-3, beta=2, kappa=0 (Merwe scaled).
// Default state: mu = 0, Sigma = I.
// update(z) returns NIS (normalized innovation squared).
template <MotionModel Motion, ObservationModel Obs>
requires std::same_as<typename Motion::StateSpace, typename Obs::StateSpace>
class UKF {
 public:
  using StateSpace = typename Motion::StateSpace;
  using MeasSpace = typename Obs::MeasSpace;
  static constexpr int N = StateSpace::state_dim;
  static constexpr int M = MeasSpace::state_dim;
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

// Compile-time proof that UKF<PosVxyzYawCV, PosYawObs> satisfies the Filter
// concept. The motion/observation headers are included here (not at the top of
// this file) to avoid a circular-include cycle: ukf.hpp -> motion.hpp ->
// state_spaces.hpp is fine, but motion.hpp already works without ukf.hpp.
#include <immtrack/motion.hpp>
#include <immtrack/observations.hpp>
namespace immtrack {
static_assert(Filter<UKF<PosVxyzYawCV, PosYawObs>>,
              "UKF<PosVxyzYawCV, PosYawObs> must satisfy Filter concept");
}  // namespace immtrack
