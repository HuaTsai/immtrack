# Tracker Architecture Redesign — 4-Axis Concept Design + IMM

**Date:** 2026-05-05
**Status:** Draft
**Supersedes / Extends:**

- `2026-05-04-bbox-tracker-design.md` (BBoxTracker shape kept; IMM moved from non-goal to first-class)
- `2026-05-04-immtrack-initial-ukf-design.md` (UKF API kept; gains concept constraints)

## Goal

Refactor the existing tracker stack onto an explicit four-axis concept
design so that:

1. State geometry, dynamics, measurement, and estimator type are
   independent and replaceable axes.
2. IMM (Interacting Multiple Model) becomes a first-class `Filter`
   variant that nests cleanly inside the existing `Track` and
   `BBoxTracker` shells.
3. Compile-time `static_assert`s catch contract breaks at the header
   that defines them, not 50 lines deep in an Eigen template
   instantiation.
4. Future state spaces / filter variants slot in without touching
   downstream code (`Track`, `BBoxTracker`, cost policies, lifecycle).

## Non-goals

- Re-identification (still scoped out as in 2026-05-04).
- Multi-class association.
- Multi-sensor fusion (different `Obs` types feeding the same tracker).
  First version stays single-`Obs`. A `std::variant<Obs...>` dispatch
  path is a Phase 3 concern.
- Quaternion / SE(3) state spaces. Ground vehicle tracking does not
  need them. The concept layer keeps the door open via `boxplus` /
  `boxminus` but no concrete non-Euclidean state space ships in MVP.
- Runtime-configurable IMM mode count. `IMM<Fs...>` is a variadic
  template — modes are fixed at compile time. Type-erased runtime IMM
  is a Phase 3 concern if it ever materialises.

## The Four Axes

| Axis              | Concept                                                                  | Examples (concrete types)        |
| ----------------- | ------------------------------------------------------------------------ | -------------------------------- |
| 1. State geometry | `Manifold`                                                               | `XYZVxVyVzYawYawRateSpace` (8D)  |
| 2. Dynamics       | `Motion` (siblings: `LinearizableMotion`, `LinearMotion`)                | `CV`, `CTRV`                     |
| 3. Measurement    | `Observation` (siblings: `LinearizableObservation`, `LinearObservation`) | `PosYawObs`                      |
| 4. Estimator      | `Filter`                                                                 | `KF`, `EKF`, `UKF`, `IMM<Fs...>` |

`Manifold` is the concept satisfied by anything with `boxplus` / `boxminus` —
both the state space and the measurement space of a filter are Manifolds. The
concrete types that satisfy it are conventionally named with a `Space` suffix
(`XYZVxVyVzYawYawRateSpace`, `PosYawMeasSpace`) and exposed via the
`StateSpace` / `MeasSpace` member typedefs of `Motion` / `Observation` /
`Filter`.

The concepts form a sibling refinement (not a single chain — `LinearMotion`
and `LinearizableMotion` are independent, since KF uses only `F_matrix` and
EKF uses only `F_jacobian`):

```text
Manifold          (boxplus / boxminus)               — any space the filter touches
  |
  +-- (every concrete StateSpace / MeasSpace satisfies Manifold)
  |
  +-- Motion                          (predict, process_noise)   — UKF / CKF / IF
  |     ├── LinearizableMotion        (+ F_jacobian)             — EKF
  |     └── LinearMotion              (+ F_matrix)               — KF
  |
  +-- Observation                     (h, measurement_noise)     — UKF / CKF
        ├── LinearizableObservation   (+ H_jacobian)             — EKF
        └── LinearObservation         (+ H_matrix)               — KF

Filter   (predict, update, state, covariance) — uniform external API
```

Naming rationale: each name describes a **mathematical property** (linearizable
at a point / linear everywhere). The siblings double as the natural eligibility
gate for each filter family, so a missing member produces a compile error
pointing at the exact concept that wasn't satisfied.

Why sibling and not a chain (`LinearMotion ⊂ LinearizableMotion`)? KF's
formulas (`x = F·x`, `P = F·P·Fᵀ + Q`) never call `F_jacobian`, so a refinement
chain would force every linear model to ship a trivial `F_jacobian = F_matrix`
shim that no current filter uses. A model that wants both EKF and KF
compatibility can satisfy both concepts independently by exposing both
members.

A reference implementation of the concept layer plus a working
`LinearKF<CV, XYObs>` lives at `ref/concept_kf_demo.hpp` /
`ref/concept_kf_demo.cc`. The design here promotes that layer into the
production headers.

## State Space Strategy — Anti-Combinatorial-Explosion

The combinatorial concern around state spaces is real. Mitigation has
two parts:

### Generator template, not hand-coded variants

```cpp
namespace immtrack::detail {

// Euclidean state with angle wrapping at compile-time-listed indices.
// Replaces the need to write a dedicated StateSpace per dim/angle layout.
template <int Dim, int... AngleIdx>
struct EuclideanWithAngles {
  using Scalar  = double;
  using State   = Eigen::Matrix<double, Dim, 1>;
  using Tangent = Eigen::Matrix<double, Dim, 1>;
  using Cov     = Eigen::Matrix<double, Dim, Dim>;
  static constexpr int state_dim   = Dim;
  static constexpr int tangent_dim = Dim;

  static State boxplus(const State& x, const Tangent& dx) noexcept {
    State r = x + dx;
    ((r(AngleIdx) = wrap_pi(r(AngleIdx))), ...);
    return r;
  }
  static Tangent boxminus(const State& a, const State& b) noexcept {
    Tangent r = a - b;
    ((r(AngleIdx) = wrap_pi(r(AngleIdx))), ...);
    return r;
  }
};

}  // namespace immtrack::detail
```

### MVP catalog: one named StateSpace

```cpp
namespace immtrack {

struct XYZVxVyVzYawYawRateSpace : detail::EuclideanWithAngles<8, /*yaw=*/6> {
  enum : int { X=0, Y=1, Z=2, VX=3, VY=4, VZ=5, YAW=6, YAW_RATE=7 };
};

}  // namespace immtrack
```

Rationale for 8D as MVP:

- Existing implementation already tracks `(x, y, z, vx, vy, vz, yaw)`
  → 7D. Going to 8D is **adding one dimension** (yaw_rate), not
  redesigning state.
- yaw_rate is required to host CTRV motion, which is the second mode
  IMM needs alongside CV.
- nuScenes / AMOTA evaluation works on full 3D bboxes; dropping z to
  go 6D BEV would regress the existing tracker.

### Future state spaces — when triggered

| Trigger                                | Add                                                                                      |
| -------------------------------------- | ---------------------------------------------------------------------------------------- |
| Need explicit acceleration mode (CA)   | extend EuclideanWithAngles with extra dims, define `XYZVxVyVzAxAyAzYawYawRateSpace` etc. |
| Drone / aerial tracking                | quaternion attitude state space (hand-written, not from generator)                       |
| 2D-only sensors (e.g. radar-only test) | `XYVxVyYawYawRateSpace<6, 4>` — one-line addition                                        |

The generator template means **adding a state space is a one-line
declaration**. There is no per-state-space implementation file to
maintain.

## Concrete API Surface (MVP)

### Headers introduced

```text
cpp/include/immtrack/
  concepts.hpp                 NEW   Manifold / Motion / Observation / Filter
  state_spaces.hpp             NEW   XYZVxVyVzYawYawRateSpace
  detail/euclidean.hpp         NEW   EuclideanWithAngles<Dim, AngleIdx...>
  motion.hpp                   MOD   CV (rewritten over StateSpace), CTRV (new)
  observations.hpp             MOD   PosYawObs (rewritten over StateSpace)
  ukf.hpp                      MOD   UKF<Mot, Obs> + concept constraints
  imm.hpp                      NEW   IMM<Filter...>
  tracker.hpp                  MOD   BBoxTracker accepts any Filter (IMM included)
```

### Filter contract

```cpp
namespace immtrack {

template <class F>
concept Filter =
    requires { typename F::StateSpace; typename F::MeasSpace; } &&
    Manifold<typename F::StateSpace> &&
    Manifold<typename F::MeasSpace>  &&
    requires(F& f, double dt,
             const typename F::MeasSpace::State& z,
             const F& cf) {
      { f.predict(dt) } -> std::same_as<void>;
      { f.update(z)   } -> std::convertible_to<double>;          // returns NIS
      { cf.state()      } -> std::convertible_to<const typename F::StateSpace::State&>;
      { cf.covariance() } -> std::convertible_to<const typename F::StateSpace::Cov&>;
    };

}  // namespace immtrack
```

### IMM as a Filter

```cpp
namespace immtrack {

template <Filter... Fs>
  requires (sizeof...(Fs) >= 2) &&
           // All sub-filters share one StateSpace (mixing requirement).
           (std::same_as<typename Fs::StateSpace,
                         typename std::tuple_element_t<0, std::tuple<Fs...>>::StateSpace>
            && ...) &&
           // All sub-filters share one MeasSpace (single-sensor IMM).
           (std::same_as<typename Fs::MeasSpace,
                         typename std::tuple_element_t<0, std::tuple<Fs...>>::MeasSpace>
            && ...)
class IMM {
 public:
  using StateSpace = typename std::tuple_element_t<0, std::tuple<Fs...>>::StateSpace;
  using MeasSpace  = typename std::tuple_element_t<0, std::tuple<Fs...>>::MeasSpace;

  IMM(std::tuple<Fs...> filters,
      Eigen::VectorXd  initial_mode_probs,
      Eigen::MatrixXd  transition_matrix);

  void   predict(double dt);
  double update(const typename MeasSpace::State& z);
  const typename StateSpace::State& state() const noexcept;
  const typename StateSpace::Cov&   covariance() const noexcept;
};

static_assert(Filter<IMM<UKF<CV, PosYawObs>, UKF<CTRV, PosYawObs>>>);

}  // namespace immtrack
```

Key invariants the constraints enforce at compile time:

- IMM must have ≥ 2 sub-filters (1 mode is not IMM).
- All sub-filters share the same `StateSpace` (mixing prerequisite).
- All sub-filters share the same `MeasSpace` (single-sensor IMM).
- Sub-filters can be **different `Filter` types** (e.g. `KF<CV>` mixed
  with `UKF<CTRV>`) as long as the above hold.
- `IMM` itself satisfies `Filter`, so `IMM<IMM<...>, ...>` (hierarchical)
  and `Track<IMM<...>>` work without changes elsewhere.

### Tracker (essentially unchanged shape)

```cpp
namespace immtrack {

template <Filter F, template <class> class CostPolicy /*, LifecyclePolicy Life*/>
class BBoxTracker {
  // Body stays close to current tracker.hpp.
  // Track<F> works as before; F is now allowed to be IMM<...>.
};

}  // namespace immtrack
```

A future `LifecyclePolicy` template parameter (M-of-N confirmation,
max-miss death) is sketched but not part of MVP — current tracker
already has hardcoded lifecycle logic that can stay.

## Phasing

### Phase 1 — Concept layer + IMM-ready single-mode (2-3 days)

1. Add `concepts.hpp` and `detail/euclidean.hpp`.
2. Add `state_spaces.hpp` with `XYZVxVyVzYawYawRateSpace`.
3. Refactor `PosVxyzYawCV` → split into:
   - `StateSpace`: `XYZVxVyVzYawYawRateSpace` (8D, gains yaw_rate).
   - Motion: `CV` over that StateSpace (yaw_rate frozen via process noise).
4. Refactor `PosYawObs` to express its `StateSpace` / `MeasSpace`
   typedefs over the new types.
5. Refactor `UKF` to constrain its parameters via `Motion` /
   `Observation`.
6. Add `static_assert(Filter<UKF<CV, PosYawObs>>)` in `ukf.hpp`.
7. Existing tests pass with one-dimension state extension (yaw_rate
   initialised to 0 with high P0 entry).

**Deliverable:** identical AMOTA scores to current `main`, but on the
new concept-constrained API. No IMM yet.

### Phase 2 — IMM (3-5 days)

1. Add `motion.hpp::CTRV` (nonlinear, requires UKF).
2. Add `imm.hpp::IMM<Filter...>` with mixing / mode-probability update.
3. Add `static_assert(Filter<IMM<UKF<CV, PosYawObs>, UKF<CTRV, PosYawObs>>>)`.
4. Wire through `BBoxTracker`. No tracker-side code changes if `Track<F>`
   only relies on the `Filter` concept — verify.
5. Benchmark CV-only vs CV+CTRV IMM on AMOTA.

**Deliverable:** IMM optionally available as the filter axis. CV-only
remains the default until benchmarking justifies the CTRV mode.

### Phase 3 — Optional (not committed)

- Mixed-filter IMM (`IMM<KF<CV>, UKF<CTRV>>`) for CV-mode performance.
- `LifecyclePolicy` extraction from `BBoxTracker`.
- CA motion + acceleration state space.
- Multi-sensor `std::variant<Obs...>` dispatch.

## Migration Impact (existing code)

| File                                     | Change                                                                                                                                           |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `cpp/include/immtrack/motion.hpp`        | `PosVxyzYawCV` → split into StateSpace + `CV<XYZVxVyVzYawYawRateSpace>`. Renames may break existing call sites; add type aliases for transition. |
| `cpp/include/immtrack/observations.hpp`  | `PosYawObs` adds `using StateSpace = ...; using MeasSpace = ...;` typedefs. State-side dimensions go from 7 to 8 (gains yaw_rate column = 0).    |
| `cpp/include/immtrack/ukf.hpp`           | Add `requires` clauses; otherwise unchanged.                                                                                                     |
| `cpp/include/immtrack/tracker.hpp`       | If `Track<Filter>` and `BBoxTracker<Filter, ...>` only call the `Filter` concept's contract, no change needed. Verify.                           |
| `cpp/include/immtrack/cost_policies.hpp` | `MahalanobisCost<Filter>` already works on filter state/cov; needs to be checked against the new `Filter::StateSpace` typedef path.              |
| `tests/`                                 | Update fixtures that hand-construct 7D states to construct 8D states (insert `yaw_rate=0`).                                                      |
| `bindings/`                              | If Python bindings expose state vector size, update to 8D.                                                                                       |
| `ref/concept_kf_demo.{hpp,cc}`           | Pedagogical demo that sketched the concept layer. Removed once Phase 1 promoted the design into production headers.                              |

## Risks

- **Phase 1 numerical equivalence.** Adding yaw_rate as a frozen
  dimension changes `Q`, `P`, sigma point count for UKF. Need to verify
  AMOTA does not regress before declaring Phase 1 done. Mitigation:
  zero-out the yaw_rate row/column of `Q` and set `P0(YAW_RATE,
YAW_RATE)` to a small value during Phase 1; CTRV in Phase 2 sets
  proper noise.
- **UKF sigma point cost.** 8D state means 17 sigma points instead of 15. Roughly 13% more cost per UKF cycle. Acceptable for tracking
  workloads; monitor.
- **Concept compile-time error message quality.** Subsumption errors
  can still be noisy if concepts are not staircased. Mitigation: per
  the demo, concepts are layered (`Motion` → `LinearizableMotion`
  → `LinearMotion`) so failures point at the exact missing member.
- **IMM mixing of yaw across modes.** When CTRV updates yaw, mixing
  with CV (which leaves yaw untouched between predicts) needs the
  shared StateSpace's `boxplus` / `boxminus` to handle yaw wrap. The
  `EuclideanWithAngles` template gives this for free; verify tests
  cover wrap edge cases.

## Open Questions

- Does `Track<F>` need to be aware of `F::StateSpace` for size /
  bbox handling, or is it purely a `Filter`-concept consumer? (Code
  read needed before Phase 1 starts.)
- Should `LifecyclePolicy` be extracted in Phase 1 or deferred? Current
  tracker has hardcoded M-of-N — leaving it hardcoded for MVP, but
  documenting it as Phase 3 keeps the door open.
- Mode probability initialisation and transition matrix: hardcode
  defaults inside `IMM` constructor, or require user to pass them?
  Leaning toward "require — there is no good default for a generic
  motion-mode mix."

## Future Extensions (Deferred)

The current `Motion` / `Observation` chain assumes **additive Gaussian noise**
(`process_noise(dt) -> Cov`, `measurement_noise() -> Cov`) and **deterministic
forward dynamics** (`predict(x, dt) -> State`). That is a deliberate scope
choice for the MVP — it covers KF / EKF / UKF / CKF / Information Filter / IMM,
which are the families this project actually plans to ship.

Filter families that need _more_ than this — sample-based filters, particle
flow, continuous-time SDE filters — are **not implemented now**, and the
concept layer does not yet model their requirements. They are sketched here so
the design space is visible the next time someone evaluates whether to extend.

### Why deferred

- No concrete plan in any phase commits to PF / EnKF / particle flow.
- The MVP filter axis (KF, EKF, UKF, IMM) sits entirely in the
  Gaussian-additive regime — adding sampling / likelihood concepts now would be
  YAGNI.
- Each new family also needs runtime infrastructure (RNGs, ensemble buffers,
  resampling) that has its own scope; the concepts are the cheap part.

### Sketch (when triggered)

These are sibling refinements of `Motion` / `Observation`, not extra rungs on
the existing chain. Adding them does **not** require changing existing
concepts or filters.

```text
Motion
├── LinearizableMotion       (EKF)             [shipped]
│   └── LinearMotion         (KF)              [shipped]
├── SamplableMotion          (PF / EnKF / Bootstrap PF)         [future]
└── SDEMotion                (Continuous-Discrete EKF)          [future]

Observation
├── LinearizableObservation  (EKF)             [shipped]
│   └── LinearObservation    (KF)              [shipped]
├── LikelihoodObservation    (PF)                                [future]
│   └── DifferentiableLikelihood  (Particle Flow / Daum-Huang)  [future]
└── ...
```

Concept signatures (illustrative — not part of the current header):

```cpp
// Forward sampling — needed when noise is non-additive or non-Gaussian.
template <class M>
concept SamplableMotion = Motion<M> && requires(State x, Scalar dt, RNG& rng) {
  { M::sample(x, dt, rng) } -> std::same_as<State>;
};

// Likelihood evaluation — replaces the Gaussian innovation assumption.
template <class O>
concept LikelihoodObservation = Observation<O> && requires(MeasState z, State x) {
  { O::log_likelihood(z, x) } -> std::convertible_to<Scalar>;
};

// Continuous-time dynamics — drift + diffusion of an SDE.
template <class M>
concept SDEMotion = requires(State x) {
  requires Manifold<typename M::StateSpace>;
  { M::drift(x) }     -> /* tangent */;
  { M::diffusion(x) } -> /* tangent x noise_dim */;
};

// Gradient of log-likelihood — needed by particle flow / Daum-Huang.
template <class O>
concept DifferentiableLikelihood = LikelihoodObservation<O> && requires(MeasState z, State x) {
  { O::log_likelihood_grad(z, x) } -> std::same_as<Tangent>;
};
```

### Filter family mapping

| Filter family                    | Required concept(s)                                 | Status                |
| -------------------------------- | --------------------------------------------------- | --------------------- |
| KF                               | `LinearMotion` + `LinearObservation`                | shipped               |
| EKF / IEKF                       | `LinearizableMotion` + `LinearizableObservation`    | shipped               |
| UKF / CKF / SRUKF                | `Motion` + `Observation`                            | shipped               |
| Information Filter / Extended IF | (same as KF / EKF in dual form)                     | shipped               |
| RTS / URTS smoother              | (same as the matching forward filter)               | shipped               |
| GSF / MHT / IMM                  | (a bank of the above; concept unchanged)            | shipped (IMM Phase 2) |
| Bootstrap / Aux PF               | `SamplableMotion` + `LikelihoodObservation`         | **deferred**          |
| EnKF                             | `SamplableMotion` + `Observation`                   | **deferred**          |
| Particle Flow / Daum-Huang       | `SamplableMotion` + `DifferentiableLikelihood`      | **deferred**          |
| Continuous-Discrete EKF          | `SDEMotion` + `LinearizableObservation`             | **deferred**          |
| Rao-Blackwellized PF             | mix of `SamplableMotion` + `LinearMotion` sub-state | **deferred**          |

### Triggers for revisiting

Reopen this section if any of the following becomes a real requirement:

- Heavy-tailed / multi-modal posterior tracking (typical justification for PF).
- Sensor models with non-Gaussian noise that can't be reasonably moment-matched.
- Continuous-time process models (e.g. tightly-coupled IMU integration where
  discretisation is the dominant error source).
- Geophysical / large-state-space tracking where ensemble methods (EnKF) beat
  Gaussian filters.

Until then this stays a paper design.

## References

- `ref/concept_kf_demo.hpp` — working concept-layer prototype.
- `ref/concept_kf_demo.cc` — runtime demo against synthetic data.
- `docs/superpowers/specs/2026-05-04-bbox-tracker-design.md` — current
  tracker shape.
- `docs/superpowers/specs/2026-05-04-immtrack-initial-ukf-design.md` —
  current UKF shape.
