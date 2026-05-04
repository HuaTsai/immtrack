#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <functional>
#include <immtrack/bbox.hpp>
#include <immtrack/detail/hungarian.hpp>
#include <immtrack/detail/iou3d.hpp>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace immtrack::metrics {

enum class MatchMetric { CenterDistance, Iou3d };

struct AmotaConfig {
  std::vector<double> recall_values = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  MatchMetric metric = MatchMetric::CenterDistance;
  double match_threshold = 2.0;
};

struct AmotaResult {
  double overall = 0.0;
  std::unordered_map<std::string, double> per_class;
};

namespace detail_amota {

inline double pair_cost(const BoundingBox &a, const BoundingBox &b, MatchMetric metric) {
  if (metric == MatchMetric::CenterDistance) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }
  return 1.0 - immtrack::detail::iou3d(a, b);
}

struct CountResult {
  int fp = 0;
  int fn = 0;
  int ids = 0;
  int gt_count = 0;
};

inline CountResult count_for_cutoff(const std::vector<std::vector<BoundingBox>> &gt_frames,
                                    const std::vector<std::vector<BoundingBox>> &pred_frames,
                                    const std::string &cls, double score_cutoff, MatchMetric metric,
                                    double match_threshold) {
  CountResult r;

  std::unordered_map<int, int> prev_match;

  for (std::size_t f = 0; f < gt_frames.size(); ++f) {
    std::vector<const BoundingBox *> gts;
    for (const auto &g : gt_frames[f]) {
      if (g.class_name == cls) {
        gts.push_back(&g);
      }
    }

    std::vector<const BoundingBox *> preds;
    if (f < pred_frames.size()) {
      for (const auto &p : pred_frames[f]) {
        if (p.class_name == cls && p.score >= score_cutoff) {
          preds.push_back(&p);
        }
      }
    }

    r.gt_count += static_cast<int>(gts.size());

    if (gts.empty()) {
      r.fp += static_cast<int>(preds.size());
      continue;
    }
    if (preds.empty()) {
      r.fn += static_cast<int>(gts.size());
      prev_match.clear();
      continue;
    }

    Eigen::MatrixXd cost(gts.size(), preds.size());
    for (std::size_t i = 0; i < gts.size(); ++i) {
      for (std::size_t j = 0; j < preds.size(); ++j) {
        const double c = pair_cost(*gts[i], *preds[j], metric);
        cost(static_cast<int>(i), static_cast<int>(j)) =
            (c > match_threshold) ? immtrack::detail::kInfeasible : c;
      }
    }

    const auto pairs = immtrack::detail::hungarian(cost);
    std::unordered_set<int> matched_gt;
    std::unordered_set<int> matched_pred;
    std::unordered_map<int, int> this_match;

    for (const auto &[i, j] : pairs) {
      if (cost(i, j) >= immtrack::detail::kInfeasible) {
        continue;
      }

      matched_gt.insert(i);
      matched_pred.insert(j);

      const int gt_id = gts[i]->track_id;
      const int pred_id = preds[j]->track_id;
      this_match[gt_id] = pred_id;

      const auto prev = prev_match.find(gt_id);
      if (prev != prev_match.end() && prev->second != pred_id) {
        r.ids += 1;
      }
    }

    r.fp += static_cast<int>(preds.size() - matched_pred.size());
    r.fn += static_cast<int>(gts.size() - matched_gt.size());
    prev_match = std::move(this_match);
  }

  return r;
}

}  // namespace detail_amota

inline AmotaResult amota(const std::vector<std::vector<BoundingBox>> &gt_frames,
                         const std::vector<std::vector<BoundingBox>> &pred_frames,
                         const AmotaConfig &cfg = {}) {
  AmotaResult result;

  std::unordered_set<std::string> classes;
  std::unordered_map<std::string, int> gt_count;
  for (const auto &frame : gt_frames) {
    for (const auto &g : frame) {
      classes.insert(g.class_name);
      gt_count[g.class_name] += 1;
    }
  }

  if (classes.empty() || cfg.recall_values.empty()) {
    return result;
  }

  double weighted_sum = 0.0;
  int total_gt = 0;

  for (const auto &cls : classes) {
    std::vector<double> scores;
    for (const auto &frame : pred_frames) {
      for (const auto &p : frame) {
        if (p.class_name == cls) {
          scores.push_back(p.score);
        }
      }
    }

    std::sort(scores.begin(), scores.end(), std::greater<double>());
    scores.erase(std::unique(scores.begin(), scores.end()), scores.end());

    const int gt_n = gt_count[cls];
    if (gt_n == 0) {
      continue;
    }

    double mota_sum = 0.0;
    for (double target_recall : cfg.recall_values) {
      bool found = false;
      double best_cutoff = std::numeric_limits<double>::infinity();

      for (double cutoff : scores) {
        const auto cr = detail_amota::count_for_cutoff(gt_frames, pred_frames, cls, cutoff,
                                                       cfg.metric, cfg.match_threshold);
        const int tp = cr.gt_count - cr.fn;
        const double recall =
            (cr.gt_count == 0) ? 0.0 : static_cast<double>(tp) / static_cast<double>(cr.gt_count);
        if (recall >= target_recall) {
          found = true;
          best_cutoff = cutoff;
          break;
        }
      }

      double mota = 0.0;
      if (found) {
        const auto cr = detail_amota::count_for_cutoff(gt_frames, pred_frames, cls, best_cutoff,
                                                       cfg.metric, cfg.match_threshold);
        const double bad = static_cast<double>(cr.fp + cr.fn + cr.ids);
        const double denom = static_cast<double>(std::max(1, cr.gt_count));
        mota = std::max(0.0, 1.0 - bad / denom);
      }
      mota_sum += mota;
    }

    const double cls_amota = mota_sum / static_cast<double>(cfg.recall_values.size());
    result.per_class[cls] = cls_amota;
    weighted_sum += cls_amota * static_cast<double>(gt_n);
    total_gt += gt_n;
  }

  if (total_gt > 0) {
    result.overall = weighted_sum / static_cast<double>(total_gt);
  }
  return result;
}

}  // namespace immtrack::metrics
