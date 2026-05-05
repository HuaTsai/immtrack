#pragma once

#include <cmath>
#include <numbers>

namespace immtrack::detail {

// Wrap angle to [-pi, pi] using IEEE 754 remainder semantics.
inline double wrap_angle(double a) noexcept { return std::remainder(a, 2.0 * std::numbers::pi); }

}  // namespace immtrack::detail
