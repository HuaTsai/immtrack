#pragma once

#include <Eigen/Core>

namespace immtrack {

// 3D bounding box observation:
//   [x, y, z, yaw, l, w, h]   (M = 7)
// Assumes the motion model places these 7 components at the front of its state
// (see motion.hpp for the convention).
struct BBox3DObs {
  static constexpr int M = 7;
  using Meas = Eigen::Matrix<double, M, 1>;

  template <class State>
  static Meas h(const State &x) {
    return x.template head<M>();
  }
};

}  // namespace immtrack
