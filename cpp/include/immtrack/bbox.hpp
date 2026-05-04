#pragma once

#include <string>

namespace immtrack {

// 3D bounding box used for both detector input and AMOTA evaluation.
// At detection time `track_id` is left at -1; for AMOTA evaluation the
// caller fills it with the GT track ID (for ground truth) or the
// tracker-assigned ID (for predictions).
struct BoundingBox {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double l = 0.0;
  double w = 0.0;
  double h = 0.0;
  double rot = 0.0;
  std::string class_name;
  double score = 0.0;
  int track_id = -1;
};

}  // namespace immtrack
