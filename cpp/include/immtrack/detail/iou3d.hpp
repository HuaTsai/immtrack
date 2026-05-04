#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <immtrack/bbox.hpp>
#include <vector>

namespace immtrack::detail {

// Sutherland-Hodgman polygon clipping for two convex CCW polygons.
inline std::vector<std::array<double, 2>> sh_clip(const std::vector<std::array<double, 2>> &subject,
                                                  const std::vector<std::array<double, 2>> &clip) {
  std::vector<std::array<double, 2>> out = subject;
  for (std::size_t i = 0; i < clip.size(); ++i) {
    if (out.empty()) {
      break;
    }
    const auto &a = clip[i];
    const auto &b = clip[(i + 1) % clip.size()];
    const double ex = b[0] - a[0];
    const double ey = b[1] - a[1];

    std::vector<std::array<double, 2>> input;
    input.swap(out);
    for (std::size_t k = 0; k < input.size(); ++k) {
      const auto &p = input[k];
      const auto &q = input[(k + 1) % input.size()];
      const double dp = ex * (p[1] - a[1]) - ey * (p[0] - a[0]);
      const double dq = ex * (q[1] - a[1]) - ey * (q[0] - a[0]);
      const bool p_in = dp >= 0.0;
      const bool q_in = dq >= 0.0;
      if (p_in) {
        out.push_back(p);
      }
      if (p_in != q_in) {
        const double t = dp / (dp - dq);
        out.push_back({p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])});
      }
    }
  }
  return out;
}

inline double polygon_area(const std::vector<std::array<double, 2>> &poly) {
  if (poly.size() < 3) {
    return 0.0;
  }
  double s = 0.0;
  for (std::size_t i = 0; i < poly.size(); ++i) {
    const auto &p = poly[i];
    const auto &q = poly[(i + 1) % poly.size()];
    s += p[0] * q[1] - q[0] * p[1];
  }
  return std::abs(s) * 0.5;
}

inline std::vector<std::array<double, 2>> bev_corners(const BoundingBox &b) {
  const double c = std::cos(b.rot);
  const double s = std::sin(b.rot);
  const double hl = b.l * 0.5;
  const double hw = b.w * 0.5;

  const std::array<std::array<double, 2>, 4> local = {
      {{{hl, hw}}, {{-hl, hw}}, {{-hl, -hw}}, {{hl, -hw}}}};

  std::vector<std::array<double, 2>> world(4);
  for (int i = 0; i < 4; ++i) {
    world[i][0] = b.x + c * local[i][0] - s * local[i][1];
    world[i][1] = b.y + s * local[i][0] + c * local[i][1];
  }
  return world;
}

inline double iou3d(const BoundingBox &a, const BoundingBox &b) {
  const auto ca = bev_corners(a);
  const auto cb = bev_corners(b);
  const auto inter_poly = sh_clip(ca, cb);
  const double inter_area = polygon_area(inter_poly);

  const double a_lo = a.z - a.h * 0.5;
  const double a_hi = a.z + a.h * 0.5;
  const double b_lo = b.z - b.h * 0.5;
  const double b_hi = b.z + b.h * 0.5;
  const double z_inter = std::max(0.0, std::min(a_hi, b_hi) - std::max(a_lo, b_lo));

  const double inter_vol = inter_area * z_inter;
  const double vol_a = a.l * a.w * a.h;
  const double vol_b = b.l * b.w * b.h;
  const double union_vol = vol_a + vol_b - inter_vol;
  if (union_vol <= 0.0) {
    return 0.0;
  }
  return inter_vol / union_vol;
}

}  // namespace immtrack::detail
