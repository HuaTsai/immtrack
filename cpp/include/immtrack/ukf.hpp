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
//       static State predict(const State& x, double dt);
//       static Cov   process_noise(double dt);
//
//       template <int K>
//       static State weighted_mean(
//           const Eigen::Matrix<double, N, K>& sigmas,
//           const Eigen::Matrix<double, K, 1>& weights);
//
//       static State residual(const State& a, const State& b);
//   };
//
//   class Obs {
//       static constexpr int M;
//       using Meas  = Eigen::Matrix<double, M, 1>;
//       using Noise = Eigen::Matrix<double, M, M>;
//
//       template <class State> static Meas h(const State& x);
//       static Noise measurement_noise();
//
//       template <int K>
//       static Meas weighted_mean(
//           const Eigen::Matrix<double, M, K>& sigmas,
//           const Eigen::Matrix<double, K, 1>& weights);
//
//       static Meas residual(const Meas& a, const Meas& b);
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

    UKF() : UKF(1e-3, 2.0, 0.0) {}

    UKF(double alpha, double beta, double kappa)
        : x_(StateVec::Zero()),
          P_(StateMat::Identity()),
          weights_(detail::UnscentedWeights<N>::make(alpha, beta, kappa)) {}

    void init(const StateVec& x, const StateMat& P) {
        x_ = x;
        P_ = P;
    }

    void predict(double dt) {
        if (dt < 0.0) {
            throw InvalidArgument("UKF::predict: dt must be non-negative");
        }

        const auto sigmas =
            detail::generate_sigma_points<N>(x_, P_, weights_.lambda);

        Eigen::Matrix<double, N, K> propagated;
        for (int i = 0; i < K; ++i) {
            propagated.col(i) = Motion::predict(sigmas.col(i), dt);
        }

        const StateVec x_new =
            Motion::template weighted_mean<K>(
                propagated, weights_.mean_weights);

        StateMat P_new = StateMat::Zero();
        for (int i = 0; i < K; ++i) {
            const StateVec d = Motion::residual(propagated.col(i), x_new);
            P_new.noalias() += weights_.cov_weights(i) * d * d.transpose();
        }
        P_new += Motion::process_noise(dt);

        x_ = x_new;
        P_ = 0.5 * (P_new + P_new.transpose());
    }

    double update(const MeasVec& z) {
        const auto sigmas =
            detail::generate_sigma_points<N>(x_, P_, weights_.lambda);

        Eigen::Matrix<double, M, K> z_sigmas;
        for (int i = 0; i < K; ++i) {
            z_sigmas.col(i) = Obs::h(sigmas.col(i));
        }

        const MeasVec z_pred =
            Obs::template weighted_mean<K>(z_sigmas, weights_.mean_weights);

        using MeasMat = Eigen::Matrix<double, M, M>;
        using CrossMat = Eigen::Matrix<double, N, M>;

        MeasMat S = MeasMat::Zero();
        CrossMat T = CrossMat::Zero();
        for (int i = 0; i < K; ++i) {
            const MeasVec dz = Obs::residual(z_sigmas.col(i), z_pred);
            const StateVec dx = Motion::residual(sigmas.col(i), x_);
            S.noalias() += weights_.cov_weights(i) * dz * dz.transpose();
            T.noalias() += weights_.cov_weights(i) * dx * dz.transpose();
        }
        S += Obs::measurement_noise();

        Eigen::LDLT<MeasMat> S_ldlt(S);
        if (S_ldlt.info() != Eigen::Success) {
            throw NumericalError(
                "UKF::update: innovation covariance not invertible");
        }

        const CrossMat K_gain = S_ldlt.solve(T.transpose()).transpose();
        const MeasVec innovation = Obs::residual(z, z_pred);

        x_.noalias() += K_gain * innovation;
        const StateMat P_post = P_ - K_gain * S * K_gain.transpose();
        P_ = 0.5 * (P_post + P_post.transpose());

        return innovation.dot(S_ldlt.solve(innovation));
    }

    const StateVec& state() const noexcept { return x_; }
    const StateMat& covariance() const noexcept { return P_; }

   private:
    StateVec x_;
    StateMat P_;
    detail::UnscentedWeights<N> weights_;
};

}  // namespace immtrack
