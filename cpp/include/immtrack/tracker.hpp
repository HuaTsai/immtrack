#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <immtrack/bbox.hpp>
#include <immtrack/detail/hungarian.hpp>
#include <immtrack/tracked_object.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace immtrack {

template <class Filter>
class Track {
 public:
  enum class Status { Tentative, Confirmed, Deleted };

  Track(int id, const BoundingBox &d)
      : id_(id), class_name_(d.class_name), size_(d.l, d.w, d.h), score_(d.score) {
    typename Filter::StateVec s;
    s.setZero();
    s(0) = d.x;
    s(1) = d.y;
    s(2) = d.z;
    s(6) = d.rot;
    filter_.init(s, Filter::StateMat::Identity());
  }

  void predict(double dt) {
    filter_.predict(dt);
    ++age_;
  }

  void update(const BoundingBox &d, double size_ema_alpha, int n_init) {
    typename Filter::MeasVec z;
    z << d.x, d.y, d.z, d.rot;
    filter_.update(z);

    // EMA size update: alpha * new + (1 - alpha) * old
    const Eigen::Vector3d obs(d.l, d.w, d.h);
    size_ = size_ema_alpha * obs + (1.0 - size_ema_alpha) * size_;

    score_ = d.score;
    ++hit_count_;
    miss_count_ = 0;

    if (status_ == Status::Tentative && hit_count_ >= n_init) {
      status_ = Status::Confirmed;
    }
  }

  void mark_missed(int max_age) {
    ++miss_count_;
    if (miss_count_ > max_age) {
      status_ = Status::Deleted;
    }
  }

  int id() const noexcept { return id_; }
  const std::string &class_name() const noexcept { return class_name_; }
  Status status() const noexcept { return status_; }
  int age() const noexcept { return age_; }
  int hit_count() const noexcept { return hit_count_; }
  int miss_count() const noexcept { return miss_count_; }
  double score() const noexcept { return score_; }
  Eigen::Vector3d size() const noexcept { return size_; }
  const Filter &filter() const noexcept { return filter_; }
  Filter &filter() noexcept { return filter_; }

  TrackedObject snapshot() const {
    const auto &x = filter_.state();
    TrackedObject t;
    t.id = id_;
    t.class_name = class_name_;
    t.x = x(0);
    t.y = x(1);
    t.z = x(2);
    t.vx = x(3);
    t.vy = x(4);
    t.vz = x(5);
    t.rot = x(6);
    t.l = size_(0);
    t.w = size_(1);
    t.h = size_(2);
    t.score = score_;
    t.age = age_;
    t.hit_count = hit_count_;
    t.miss_count = miss_count_;
    return t;
  }

 private:
  Filter filter_;
  int id_;
  std::string class_name_;
  Eigen::Vector3d size_;
  double score_;
  int hit_count_ = 1;
  int miss_count_ = 0;
  int age_ = 0;
  Status status_ = Status::Tentative;
};

template <class Filter, template <class> class CostPolicy>
class BBoxTracker {
 public:
  struct Config {
    int n_init = 3;
    int max_age = 5;
    int min_hits = 3;
    double size_ema_alpha = 0.7;
  };

  explicit BBoxTracker(Config cfg = {}) : cfg_(cfg) {}

  std::vector<TrackedObject> update(const std::vector<BoundingBox> &detections) {
    return update_impl(detections, /*has_dt=*/false, /*dt=*/0.0);
  }

  std::vector<TrackedObject> update(const std::vector<BoundingBox> &detections, double dt) {
    return update_impl(detections, /*has_dt=*/true, dt);
  }

  void reset() {
    tracks_.clear();
    next_id_ = 0;
  }

  std::size_t track_count() const noexcept { return tracks_.size(); }

 private:
  using TrackT = Track<Filter>;

  std::vector<TrackedObject> update_impl(const std::vector<BoundingBox> &detections, bool has_dt,
                                         double dt) {
    if (has_dt) {
      for (auto &t : tracks_) {
        if (t.status() != TrackT::Status::Deleted) {
          t.predict(dt);
        }
      }
    }

    std::unordered_map<std::string, std::vector<int>> track_by_class;
    std::unordered_map<std::string, std::vector<int>> det_by_class;

    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
      if (tracks_[i].status() != TrackT::Status::Deleted) {
        track_by_class[tracks_[i].class_name()].push_back(i);
      }
    }
    for (int j = 0; j < static_cast<int>(detections.size()); ++j) {
      det_by_class[detections[j].class_name].push_back(j);
    }

    std::vector<bool> track_matched(tracks_.size(), false);
    std::vector<bool> det_matched(detections.size(), false);

    for (const auto &[cls, det_idxs] : det_by_class) {
      const auto trk_it = track_by_class.find(cls);
      if (trk_it == track_by_class.end() || trk_it->second.empty()) {
        continue;
      }
      const auto &trk_idxs = trk_it->second;

      const int rows = static_cast<int>(trk_idxs.size());
      const int cols = static_cast<int>(det_idxs.size());
      Eigen::MatrixXd cost(rows, cols);
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          const double v =
              CostPolicy<Filter>::cost(tracks_[trk_idxs[r]].filter(), detections[det_idxs[c]]);
          cost(r, c) = (v >= CostPolicy<Filter>::gate_threshold()) ? detail::kInfeasible : v;
        }
      }

      const auto pairs = detail::hungarian(cost);
      for (const auto &[r, c] : pairs) {
        if (cost(r, c) >= detail::kInfeasible) {
          continue;
        }
        const int trk = trk_idxs[r];
        const int det = det_idxs[c];
        tracks_[trk].update(detections[det], cfg_.size_ema_alpha, cfg_.n_init);
        track_matched[trk] = true;
        det_matched[det] = true;
      }
    }

    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
      if (!track_matched[i] && tracks_[i].status() != TrackT::Status::Deleted) {
        tracks_[i].mark_missed(cfg_.max_age);
      }
    }

    for (int j = 0; j < static_cast<int>(detections.size()); ++j) {
      if (!det_matched[j]) {
        tracks_.emplace_back(next_id_++, detections[j]);
      }
    }

    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [](const TrackT &t) { return t.status() == TrackT::Status::Deleted; }),
        tracks_.end());

    std::vector<TrackedObject> out;
    out.reserve(tracks_.size());
    for (const auto &t : tracks_) {
      if (t.status() == TrackT::Status::Confirmed && t.miss_count() == 0 &&
          t.hit_count() >= cfg_.min_hits) {
        out.push_back(t.snapshot());
      }
    }
    return out;
  }

  Config cfg_;
  std::vector<TrackT> tracks_;
  int next_id_ = 0;
};

}  // namespace immtrack
