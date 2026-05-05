#pragma once

#include <Eigen/Core>
#include <concepts>
#include <type_traits>

namespace immtrack {

// 1. StateSpace — geometry of the state manifold.
template <class S>
concept StateSpace = requires(typename S::State   x,
                              typename S::Tangent dx,
                              typename S::State   a,
                              typename S::State   b) {
  typename S::Scalar;
  typename S::State;
  typename S::Tangent;
  typename S::Cov;
  { S::state_dim   } -> std::convertible_to<int>;
  { S::tangent_dim } -> std::convertible_to<int>;
  requires std::floating_point<typename S::Scalar>;
  { S::boxplus(x, dx) } -> std::same_as<typename S::State>;
  { S::boxminus(a, b) } -> std::same_as<typename S::Tangent>;
};

// 2. MotionModel — dynamics over a StateSpace.
template <class M>
concept MotionModel =
    requires(const typename M::StateSpace::State& x,
             typename M::StateSpace::Scalar       dt) {
      requires StateSpace<typename M::StateSpace>;
      { M::predict(x, dt) }    -> std::same_as<typename M::StateSpace::State>;
      { M::process_noise(dt) } -> std::same_as<typename M::StateSpace::Cov>;
    };

template <class M>
concept HasMotionJacobian =
    MotionModel<M> &&
    requires(const typename M::StateSpace::State& x,
             typename M::StateSpace::Scalar       dt) {
      { M::F_jacobian(x, dt) } -> std::same_as<typename M::StateSpace::Cov>;
    };

template <class M>
concept LinearMotion =
    HasMotionJacobian<M> &&
    requires(typename M::StateSpace::Scalar dt) {
      { M::F_matrix(dt) } -> std::same_as<typename M::StateSpace::Cov>;
    };

// 3. ObservationModel — projection from StateSpace into MeasSpace.
template <class O>
using ObsJacMat = Eigen::Matrix<typename O::StateSpace::Scalar,
                                O::MeasSpace::tangent_dim,
                                O::StateSpace::tangent_dim>;

template <class O>
concept ObservationModel =
    requires(const typename O::StateSpace::State& x) {
      requires StateSpace<typename O::StateSpace>;
      requires StateSpace<typename O::MeasSpace>;
      requires std::same_as<typename O::StateSpace::Scalar,
                            typename O::MeasSpace::Scalar>;
      { O::h(x) } -> std::same_as<typename O::MeasSpace::State>;
      { O::measurement_noise() } -> std::same_as<typename O::MeasSpace::Cov>;
    };

template <class O>
concept HasObsJacobian =
    ObservationModel<O> &&
    requires(const typename O::StateSpace::State& x) {
      { O::H_jacobian(x) } -> std::same_as<ObsJacMat<O>>;
    };

template <class O>
concept LinearObs =
    HasObsJacobian<O> &&
    requires {
      { O::H_matrix() } -> std::same_as<ObsJacMat<O>>;
    };

// 4. Filter — uniform external API for KF / EKF / UKF / IMM.
template <class F>
concept Filter =
    requires { typename F::StateSpace; typename F::MeasSpace; } &&
    StateSpace<typename F::StateSpace> &&
    StateSpace<typename F::MeasSpace>  &&
    requires(F& f, double dt,
             const typename F::MeasSpace::State& z,
             const F& cf) {
      { f.predict(dt) } -> std::same_as<void>;
      { f.update(z)   } -> std::convertible_to<double>;
      { cf.state()      } -> std::convertible_to<const typename F::StateSpace::State&>;
      { cf.covariance() } -> std::convertible_to<const typename F::StateSpace::Cov&>;
    };

}  // namespace immtrack
