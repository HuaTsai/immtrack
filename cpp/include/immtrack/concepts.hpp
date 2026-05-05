#pragma once

#include <Eigen/Core>
#include <concepts>
#include <type_traits>

namespace immtrack {

// 1. Manifold — boxplus/boxminus geometry.
//
// The shared abstraction for any space the filter operates on: a state space,
// a measurement space, or any future product/quotient manifold. Captures
// "locally Euclidean, globally may wrap" via boxplus / boxminus
// (Hertzberg et al. 2013).
//
// Concrete uses in this codebase:
//   Motion::StateSpace, Observation::StateSpace, Filter::StateSpace — Manifold
//   Observation::MeasSpace, Filter::MeasSpace                       — Manifold
template <class M>
concept Manifold = requires(typename M::State x, typename M::Tangent dx, typename M::State a,
                            typename M::State b) {
  typename M::Cov;
  requires std::floating_point<typename M::Scalar>;
  { M::state_dim } -> std::convertible_to<int>;
  { M::tangent_dim } -> std::convertible_to<int>;
  { M::boxplus(x, dx) } -> std::same_as<typename M::State>;
  { M::boxminus(a, b) } -> std::same_as<typename M::Tangent>;
};

// 2. Motion — dynamics over a Manifold (the model's StateSpace).
//
// Sibling refinements of Motion (each filter family asks for what it actually uses):
//   Motion              — UKF / CKF / Information Filter compatible.
//   LinearizableMotion  — adds F_jacobian: EKF compatible.
//   LinearMotion        — adds F_matrix:   KF  compatible.
//
// LinearMotion is *not* a refinement of LinearizableMotion: KF formulas use
// F_matrix only, never F_jacobian. A model that wants to be usable by both KF
// and EKF can satisfy both concepts independently (F_jacobian is a trivial
// passthrough of F_matrix in that case).
//
// Future siblings (deferred — see specs/2026-05-05-tracker-architecture-redesign.md
// "Future extensions"): SamplableMotion (PF / EnKF), SDEMotion (CD-EKF).
template <class M>
concept Motion =
    requires(const typename M::StateSpace::State &x, typename M::StateSpace::Scalar dt) {
      requires Manifold<typename M::StateSpace>;
      { M::predict(x, dt) } -> std::same_as<typename M::StateSpace::State>;
      { M::process_noise(dt) } -> std::same_as<typename M::StateSpace::Cov>;
    };

template <class M>
concept LinearizableMotion =
    requires(const typename M::StateSpace::State &x, typename M::StateSpace::Scalar dt) {
      requires Motion<M>;
      { M::F_jacobian(x, dt) } -> std::same_as<typename M::StateSpace::Cov>;
    };

template <class M>
concept LinearMotion = requires(typename M::StateSpace::Scalar dt) {
  requires Motion<M>;
  { M::F_matrix(dt) } -> std::same_as<typename M::StateSpace::Cov>;
};

// 3. Observation — projection between two Manifolds (StateSpace -> MeasSpace).
//
// Sibling refinements (mirrors Motion):
//   Observation              — UKF / CKF compatible.
//   LinearizableObservation  — adds H_jacobian: EKF compatible.
//   LinearObservation        — adds H_matrix:   KF  compatible.
//
// LinearObservation is *not* a refinement of LinearizableObservation — KF
// uses H_matrix only.
//
// Future siblings (deferred): LikelihoodObservation (PF),
// DifferentiableLikelihood (Particle Flow / Daum-Huang).
template <class O>
using ObsJacMat = Eigen::Matrix<typename O::StateSpace::Scalar, O::MeasSpace::tangent_dim,
                                O::StateSpace::tangent_dim>;

template <class O>
concept Observation = requires(const typename O::StateSpace::State &x) {
  requires Manifold<typename O::StateSpace>;
  requires Manifold<typename O::MeasSpace>;
  requires std::same_as<typename O::StateSpace::Scalar, typename O::MeasSpace::Scalar>;
  { O::h(x) } -> std::same_as<typename O::MeasSpace::State>;
  { O::measurement_noise() } -> std::same_as<typename O::MeasSpace::Cov>;
};

template <class O>
concept LinearizableObservation = requires(const typename O::StateSpace::State &x) {
  requires Observation<O>;
  { O::H_jacobian(x) } -> std::same_as<ObsJacMat<O>>;
};

template <class O>
concept LinearObservation = requires {
  requires Observation<O>;
  { O::H_matrix() } -> std::same_as<ObsJacMat<O>>;
};

// 4. Filter — uniform external API for KF / EKF / UKF / IMM.
template <class F>
concept Filter =
    requires {
      typename F::StateSpace;
      typename F::MeasSpace;
    } && Manifold<typename F::StateSpace> && Manifold<typename F::MeasSpace> &&
    requires(F &f, double dt, const typename F::MeasSpace::State &z, const F &cf) {
      { f.predict(dt) } -> std::same_as<void>;
      { f.update(z) } -> std::convertible_to<double>;
      { cf.state() } -> std::convertible_to<const typename F::StateSpace::State &>;
      { cf.covariance() } -> std::convertible_to<const typename F::StateSpace::Cov &>;
    };

}  // namespace immtrack
