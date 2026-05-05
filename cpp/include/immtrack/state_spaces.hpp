#pragma once

#include <immtrack/detail/euclidean.hpp>

namespace immtrack {

// 8D state for ground-vehicle 3D bbox tracking.
// Index layout: [x, y, z, vx, vy, vz, yaw, yaw_rate].
// `yaw` (idx 6) wraps to (-pi, pi]; `yaw_rate` is plain Euclidean.
struct XYZVxVyVzYawYawRateSpace : detail::EuclideanWithAngles<8, /*yaw=*/6> {
  enum : int {
    X = 0,
    Y = 1,
    Z = 2,
    VX = 3,
    VY = 4,
    VZ = 5,
    YAW = 6,
    YAW_RATE = 7,
  };
};

}  // namespace immtrack
