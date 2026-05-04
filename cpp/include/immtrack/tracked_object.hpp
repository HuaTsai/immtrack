#pragma once

#include <string>

namespace immtrack {

// Per-track snapshot returned from BBoxTracker::update().
struct TrackedObject {
  int id = -1;
  std::string class_name;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double vx = 0.0;
  double vy = 0.0;
  double vz = 0.0;
  double rot = 0.0;
  double l = 0.0;
  double w = 0.0;
  double h = 0.0;
  double score = 0.0;
  int age = 0;
  int hit_count = 0;
  int miss_count = 0;
};

}  // namespace immtrack
