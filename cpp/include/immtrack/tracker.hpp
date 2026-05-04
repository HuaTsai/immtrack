#pragma once

#include <Eigen/Core>
#include <string>

#include <immtrack/bbox.hpp>
#include <immtrack/tracked_object.hpp>

namespace immtrack {

template <class Filter>
class Track {
   public:
    enum class Status { Tentative, Confirmed, Deleted };

    Track(int id, const BoundingBox& d)
        : id_(id),
          class_name_(d.class_name),
          size_(d.l, d.w, d.h),
          score_(d.score) {
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

    void update(const BoundingBox& d, double size_ema_alpha, int n_init) {
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
    const std::string& class_name() const noexcept { return class_name_; }
    Status status() const noexcept { return status_; }
    int age() const noexcept { return age_; }
    int hit_count() const noexcept { return hit_count_; }
    int miss_count() const noexcept { return miss_count_; }
    double score() const noexcept { return score_; }
    Eigen::Vector3d size() const noexcept { return size_; }
    const Filter& filter() const noexcept { return filter_; }
    Filter& filter() noexcept { return filter_; }

    TrackedObject snapshot() const {
        const auto& x = filter_.state();
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

}  // namespace immtrack
