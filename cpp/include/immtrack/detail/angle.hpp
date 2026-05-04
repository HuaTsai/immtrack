#pragma once

#include <cmath>

namespace immtrack::detail {

// Wrap angle to [-pi, pi]. Uses atan2(sin, cos) for robustness against
// large inputs and IEEE 754 special values; fmod is implementation-defined
// for negative operands and prone to round-off near boundaries.
inline double wrap_angle(double a) noexcept { return std::atan2(std::sin(a), std::cos(a)); }

}  // namespace immtrack::detail
