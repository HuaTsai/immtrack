#pragma once

#include <stdexcept>

#include <Eigen/Core>

namespace immtrack {

// Unscented Kalman Filter parameterised by motion model + observation traits.
// One of several filter back-ends planned for immtrack (EKF, ParticleFilter,
// ... will live alongside this header). Motion / Obs traits are filter-
// agnostic and can be reused across back-ends.
template <class Motion, class Obs>
class UKF {
 public:
  static constexpr int N = Motion::N;
  static constexpr int M = Obs::M;
  using StateVec = Eigen::Matrix<double, N, 1>;
  using StateMat = Eigen::Matrix<double, N, N>;
  using MeasVec = Eigen::Matrix<double, M, 1>;

  UKF() : x_(StateVec::Zero()), P_(StateMat::Identity()) {}

  void init(const StateVec &x, const StateMat &P) {
    x_ = x;
    P_ = P;
  }

  void predict(double dt) {
    // TODO: full sigma-point propagation + covariance update.
    // For now: propagate the mean only via Motion::predict.
    x_ = Motion::predict(x_, dt);
    P_ += Motion::process_noise(dt);
  }

  void update(const MeasVec & /*z*/) {
    // TODO: sigma-point projection through Obs::h, Kalman gain, posterior.
    throw std::runtime_error("UKF::update not implemented yet");
  }

  const StateVec &state() const { return x_; }
  const StateMat &covariance() const { return P_; }

 private:
  StateVec x_;
  StateMat P_;
};

}  // namespace immtrack
