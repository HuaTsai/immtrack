# BBoxTracker Design

**Date:** 2026-05-04
**Status:** Draft
**Depends on:** UKF implementation (`UkfPosVxyzYawCV` from 2026-05-04-immtrack-initial-ukf-design.md)

## Goal

Provide a multi-object tracker over 3D bounding boxes that wraps the existing
UKF, supports strict per-class data association, and exposes a clean
single-method `update()` API for both the first frame and subsequent frames.
Also provide an AMOTA evaluation utility (C++ implementation, exposed to
Python) for benchmarking tracker output against ground truth.

## Non-goals

- Re-identification of deleted tracks (each disappearance gets a new ID).
- Track merge / split post-processing.
- Multi-class association (across classes).
- Online appearance feature extraction (only geometry is used).
- IMM (Interacting Multiple Model) — the tracker is templated on `Filter`,
  so an IMM filter can be plugged in later without changing the tracker.

## Architecture Overview

The tracker is a header-only C++ template parameterised on:

1. `Filter` — the motion/observation filter (initially `UkfPosVxyzYawCV`).
2. `CostPolicy` — the association cost functor (initially `MahalanobisCost`,
   future: `GIoUCost`).

The Hungarian solver is hardcoded (not pluggable) since association strategy
is unlikely to change while cost functions will.

```
+-------------------------------------------------+
|                BBoxTracker<Filter, CostPolicy>  |
|                                                 |
|  +-----------+    +---------------+             |
|  | Track<F>  |    | Hungarian     |             |
|  | (per obj) |    | solver (O(n^3))|            |
|  +-----------+    +---------------+             |
|       ^                  ^                      |
|       |                  |                      |
|  predict / update    cost matrix                |
|       |                  |                      |
|       +--- CostPolicy<Filter> -----+            |
|              (Mahalanobis MVP)                  |
+-------------------------------------------------+
```

## Public API

### Data structures

```cpp
namespace immtrack {

struct BoundingBox {
    double x, y, z;            // center
    double l, w, h;            // size (length, width, height)
    double rot;                // yaw in radians
    std::string class_name;    // e.g. "car", "pedestrian"
    double score;              // detector confidence (GT may use 1.0)
    int track_id = -1;         // -1 at detection time; filled for AMOTA
                               // evaluation input (GT track ID or
                               // tracker-assigned ID for predictions)
};

struct TrackedObject {
    int id;
    std::string class_name;
    double x, y, z;            // estimated position
    double vx, vy, vz;         // estimated velocity
    double rot;                // estimated yaw
    double l, w, h;            // EMA-smoothed size
    double score;              // last observation score
    int age;                   // frames since creation
    int hit_count;             // total successful updates
    int miss_count;            // consecutive misses
};

}  // namespace immtrack
```

### Cost policy trait

```cpp
template <class Filter>
struct MahalanobisCost {
    // Returns Mahalanobis distance using filter's innovation covariance S.
    // Lower = better. Returns +inf if numerical failure.
    static double cost(const Filter& f, const BoundingBox& d);

    // χ²(4, 0.99) ≈ 13.28 — costs above this are gated out.
    static constexpr double gate_threshold() { return 13.28; }
};
```

### Tracker class

```cpp
template <class Filter, template<class> class CostPolicy = MahalanobisCost>
class BBoxTracker {
public:
    struct Config {
        int n_init = 3;             // hits to confirm a tentative track
        int max_age = 5;            // misses before deletion
        int min_hits = 3;           // hits needed to be reported (after confirm)
        double size_ema_alpha = 0.7;// EMA weight on new observation
    };

    explicit BBoxTracker(Config cfg = {});

    // First frame — no time delta needed.
    std::vector<TrackedObject> update(
        const std::vector<BoundingBox>& detections);

    // Subsequent frames.
    std::vector<TrackedObject> update(
        const std::vector<BoundingBox>& detections, double dt);

    void reset();
    std::size_t track_count() const noexcept;
};
```

### Per-track object

`Track<Filter>` is internal but documented here for clarity:

```cpp
template <class Filter>
class Track {
public:
    enum class Status { Tentative, Confirmed, Deleted };

    Track(int id, const BoundingBox& d);

    void predict(double dt);            // calls filter_.predict(dt)
    void update(const BoundingBox& d, double size_ema_alpha);
    void mark_missed();

    int id() const noexcept;
    const std::string& class_name() const noexcept;
    Status status() const noexcept;
    int age() const noexcept;
    int hit_count() const noexcept;
    int miss_count() const noexcept;
    const Filter& filter() const noexcept;
    Eigen::Vector3d size() const noexcept;
    double score() const noexcept;

    TrackedObject snapshot() const;     // assemble TrackedObject for output

private:
    Filter filter_;
    int id_;
    std::string class_name_;
    Eigen::Vector3d size_;              // EMA-smoothed (l, w, h)
    double score_;
    int hit_count_ = 1;                 // counted at construction
    int miss_count_ = 0;
    int age_ = 0;
    Status status_ = Status::Tentative;
};
```

## Algorithm

### Per-frame flow

1. **Predict** — for every non-deleted track, call `track.predict(dt)`.
   Skipped on first frame (no `dt`).
2. **Per-class bucketing** — group both tracks and detections by
   `class_name`. Each bucket is associated independently.
3. **Build cost matrix** for each bucket. Cell `(i, j)` is
   `CostPolicy::cost(track_i, det_j)`. Cells exceeding
   `CostPolicy::gate_threshold()` are replaced with `INFEASIBLE = 1e9`.
4. **Hungarian assignment** on the cost matrix. Pairs with cost ≥
   `INFEASIBLE` are rejected after the solver returns.
5. **Update matched** tracks: `track.update(detection, alpha)`,
   `hit_count++`, `miss_count = 0`, size EMA update, `status = Confirmed`
   if `hit_count >= n_init`.
6. **Unmatched tracks**: `track.mark_missed()`, `miss_count++`.
   If `miss_count > max_age`, `status = Deleted`.
7. **Unmatched detections**: spawn new `Track` in `Tentative` state with
   a freshly allocated ID. The spawning detection counts as the first
   hit (`hit_count_ = 1` at construction).
8. **Garbage collect** deleted tracks.
9. **Output**: collect `TrackedObject` snapshot for every track whose
   `status == Confirmed` and `miss_count == 0` (i.e. matched this frame).
   Tracks that are confirmed but missed this frame are kept in the
   internal pool but not reported. `min_hits` is retained in `Config` as
   an additional gate (`hit_count >= min_hits`) for cases where users
   want a stricter reporting threshold than `n_init`; with the default
   `n_init == min_hits == 3` it is redundant but harmless.

### Track state machine

```
                 ┌──────────────┐
   new det ──▶  │  Tentative   │
                 └──────┬───────┘
                        │ n_init hits
                        ▼
                 ┌──────────────┐
                 │  Confirmed   │  ◀── reported in update() output
                 └──────┬───────┘
                        │ miss_count > max_age
                        ▼
                 ┌──────────────┐
                 │   Deleted    │  ◀── (Tentative tracks also enter here
                 └──────────────┘       on first miss before confirmation)
```

### Cost policy: Mahalanobis (MVP)

For a measurement `z = [x, y, z, yaw]` and a filter that has just been
predicted (so it owns innovation covariance `S = H P Hᵀ + R`):

```
ν = z - h(x_pred)            # innovation (with yaw normalisation)
d² = νᵀ S⁻¹ ν                 # squared Mahalanobis distance
```

`d²` is the cost. Gate at χ²(4, 0.99) ≈ 13.28. The UKF must expose
the post-prediction `S` and innovation helper. (The UKF interface
already exposes the necessary primitives via `predict_measurement()`.)

### Cost policy: GIoU (future)

A future `GIoUCost<Filter>` will implement a 3D generalised IoU using
the filter's predicted center + yaw and the track's stored `(l, w, h)`.
It will be a drop-in replacement; no tracker change required.

### Hungarian solver

Header-only O(n³) implementation in
`cpp/include/immtrack/detail/hungarian.hpp`. Square or rectangular cost
matrices supported (rectangular is padded with `INFEASIBLE` internally).
Returns vector of (row, col) pairs.

Why custom and not a library:

- Per-class buckets keep N small (typically < 50). O(n³) is sufficient.
- Avoids adding a third-party dependency for a 100-line algorithm.
- Easy to unit-test against known-optimal small matrices.

### Size maintenance: EMA

Track size `(l, w, h)` is initialised from the spawning detection. On
each successful update:

```
size_new = α * observed_size + (1 - α) * size_old
```

with `α = 0.7` (configurable). Single `Eigen::Vector3d`, O(1) update.

### ID management (MVP)

- Monotonically increasing global counter, allocated at track creation.
- Deleted tracks never re-emerge; if the same physical object reappears
  after deletion, it gets a new ID.
- Hungarian + per-class strict + `n_init = 3` confirmation prevents
  same-frame splits. Tentative tracks competing for the same detection
  are resolved within the assignment matrix.
- ID reincarnation / appearance reID is explicitly out of scope.

## AMOTA Evaluation Utility

AMOTA (Average Multi-Object Tracking Accuracy, AB3DMOT formulation) is
implemented in **C++** and exposed to Python through pybind11. The
internal Hungarian solver is reused for per-frame GT↔prediction matching.

### Definitions

For a given confidence-recall threshold `r`:

```
MOTA(r) = max(0, 1 − (FP(r) + FN(r) + IDS(r)) / GT_count)
```

AMOTA averages MOTA over a discrete set of recall values:

```
AMOTA = (1 / |R|) * Σ_{r ∈ R} MOTA(r)
```

Default recall set: `R = {0.1, 0.2, ..., 0.9}` (AB3DMOT default).

Even though only AMOTA is reported, FP / FN / IDS must be counted
internally per recall threshold to evaluate MOTA(r).

### Matching criterion

A prediction is matched to a GT in a given frame iff their cost is below
a threshold. Two metrics provided:

1. **Center distance** (default) — Euclidean distance on `(x, y, z)`.
   Threshold: 2.0 m.
2. **3D IoU** — `1 - IoU` as cost. Threshold: 0.5 IoU.

Per-frame matching uses the C++ Hungarian solver from
`detail/hungarian.hpp`. Cells exceeding the threshold are gated to
`INFEASIBLE`.

### C++ API

```cpp
namespace immtrack::metrics {

enum class MatchMetric { CenterDistance, Iou3d };

struct AmotaConfig {
    std::vector<double> recall_values = {0.1, 0.2, 0.3, 0.4, 0.5,
                                          0.6, 0.7, 0.8, 0.9};
    MatchMetric metric = MatchMetric::CenterDistance;
    double match_threshold = 2.0;   // metres for CenterDistance,
                                    // (1 - IoU) for Iou3d
};

struct AmotaResult {
    double overall;                                      // weighted by GT count
    std::unordered_map<std::string, double> per_class;   // {class_name: AMOTA}
};

// Inputs are frame-major:
//   gt[f]   = ground-truth boxes for frame f (track_id required, score ignored)
//   pred[f] = predicted boxes for frame f   (track_id and score required)
AmotaResult amota(
    const std::vector<std::vector<BoundingBox>>& gt,
    const std::vector<std::vector<BoundingBox>>& pred,
    const AmotaConfig& cfg = {});

}  // namespace immtrack::metrics
```

### Python API

Exposed via pybind11 as `immtrack.metrics.amota`. C++ struct is mapped
to a Python dataclass-shaped object (read-only attributes `overall` and
`per_class`).

```python
from immtrack import BoundingBox
from immtrack.metrics import amota, AmotaConfig, MatchMetric

result = amota(gt_frames, pred_frames, AmotaConfig(
    metric=MatchMetric.CenterDistance,
    match_threshold=2.0,
))
print(result.overall)         # float
print(result.per_class)       # {"car": 0.72, "pedestrian": 0.55}
```

### Algorithm sketch

```
For each class c:
    GT_c     = all GT boxes belonging to class c (across frames)
    PRED_c   = all pred boxes belonging to class c, sorted by score desc

    For each recall r in R:
        Determine score cutoff that yields recall r over GT_c
            (binary search on sorted score list, counting TP after
             per-frame matching with full prediction set first).
        Replay matching with predictions filtered to score >= cutoff:
            For each frame f:
                Build cost matrix (GT_c[f] × PRED_filtered[f]) using
                    match_metric, gate cells > match_threshold.
                Hungarian solve.
                Accumulate TP, FP, FN.
                Compare matched pred track_id to previous-frame match
                    of same GT track_id; if differs, IDS += 1.
        MOTA(r) = max(0, 1 - (FP + FN + IDS) / |GT_c|)

    AMOTA_c = mean of MOTA(r) over R

AmotaResult.per_class[c] = AMOTA_c
AmotaResult.overall      = Σ_c (|GT_c| * AMOTA_c) / Σ_c |GT_c|
```

### Out of scope for v1 of the metric

- Time-tagged async input (assume synchronous frame-aligned input).
- Multi-camera / multi-modal handling.
- nuScenes-specific category remapping.
- Reporting sAMOTA, AMOTP, FRAG, MT, ML (future work).

## File Layout

```
cpp/include/immtrack/
├── bbox.hpp              # BoundingBox struct
├── tracked_object.hpp    # TrackedObject struct
├── tracker.hpp           # BBoxTracker, Track
├── cost_policies.hpp     # MahalanobisCost (future: GIoUCost)
├── metrics/
│   └── amota.hpp         # AmotaConfig, AmotaResult, amota()
└── detail/
    ├── hungarian.hpp     # Hungarian solver (shared by tracker + metrics)
    └── iou3d.hpp         # 3D IoU helper (used by amota Iou3d metric)

bindings/
└── _core.cc              # exports BBoxTracker, BoundingBox, TrackedObject,
                          # AmotaConfig, AmotaResult, MatchMetric, amota()

src/immtrack/
├── _core.pyi             # type stubs for above
└── metrics/
    └── __init__.py       # re-exports immtrack._core.metrics.* into
                          # immtrack.metrics.{amota, AmotaConfig, ...}

tests/
├── cpp/
│   ├── test_hungarian.cpp
│   ├── test_track_lifecycle.cpp
│   ├── test_mahalanobis_cost.cpp
│   ├── test_bbox_tracker.cpp
│   └── test_amota.cpp
└── python/
    ├── test_bbox_tracker.py
    └── test_amota.py
```

## Python Binding Shape

A single concrete instantiation is exposed:

```python
from immtrack import BoundingBox, TrackedObject, BBoxTracker

tracker = BBoxTracker()                       # default Config
tracker = BBoxTracker(n_init=2, max_age=10)   # kwargs forwarded to Config

# First frame
out = tracker.update([BoundingBox(...), ...])

# Later frames
out = tracker.update(detections, dt=0.1)
```

Internally the binding fixes `Filter = UkfPosVxyzYawCV` and
`CostPolicy = MahalanobisCost`. When future filters or cost policies are
added, additional Python class names will be exported (e.g.
`BBoxTrackerImmGIoU`).

## Testing Strategy

### C++ unit tests (Catch2)

| Test                    | Coverage                                                                                                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_hungarian`        | Square, rectangular, all-infeasible, single-cell, known-optimal small matrices                                                                                                 |
| `test_track_lifecycle`  | Tentative→Confirmed after `n_init`; Confirmed→Deleted after `max_age` misses; tentative deletion on early miss                                                                 |
| `test_mahalanobis_cost` | Cost on synthetic UKF; verify gating at χ² threshold; numerical singular `S` returns +inf                                                                                      |
| `test_bbox_tracker`     | Single object 5-frame straight line; two objects different classes (no cross-association); occlusion (1 frame missed); two new detections in same frame                        |
| `test_amota`            | Perfect predictions → AMOTA = 1.0; pure FP → AMOTA degrades; ID switch injection → IDS counted; per-class breakdown correct; CenterDistance vs Iou3d both produce sane numbers |

### Python tests (pytest)

| Test                   | Coverage                                                                                                                                      |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_bbox_tracker.py` | End-to-end 10-frame synthetic sequence with 2 cars and 1 pedestrian; assert ID continuity and final state proximity to truth                  |
| `test_amota.py`        | Smoke test that pybind11 binding round-trips Config / Result correctly; one synthetic perfect-prediction case asserts `result.overall == 1.0` |

## Open Questions

None at brainstorming time. Decisions logged inline above.

## Future Work

- IMM filter (CV + CTRA) plugged into `Filter` template parameter.
- `GIoUCost` cost policy.
- Re-ID buffer for short-occlusion ID continuity.
- Configurable per-class Config overrides (different `max_age` for
  pedestrians vs cars).
- Score-weighted EMA for size update.
- Reporting sAMOTA, AMOTP, FRAG, MT, ML alongside AMOTA.
