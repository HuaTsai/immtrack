#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <immtrack/errors.hpp>

namespace immtrack::detail {

// Merwe scaled unscented weights.
//   lambda = alpha^2 * (N + kappa) - N
//   W_m_0 = lambda / (N + lambda)
//   W_c_0 = lambda / (N + lambda) + (1 - alpha^2 + beta)
//   W_m_i = W_c_i = 1 / (2 (N + lambda))   for i = 1..2N
template <int N>
struct UnscentedWeights {
  static constexpr int K = 2 * N + 1;
  using Vec = Eigen::Matrix<double, K, 1>;

  Vec mean_weights;
  Vec cov_weights;
  double lambda;

  static UnscentedWeights make(double alpha, double beta, double kappa) {
    UnscentedWeights w{};
    w.lambda = alpha * alpha * (N + kappa) - N;
    const double denom = static_cast<double>(N) + w.lambda;
    const double rest = 1.0 / (2.0 * denom);
    w.mean_weights(0) = w.lambda / denom;
    w.cov_weights(0) = w.lambda / denom + (1.0 - alpha * alpha + beta);
    for (int i = 1; i < K; ++i) {
      w.mean_weights(i) = rest;
      w.cov_weights(i) = rest;
    }
    return w;
  }
};

// Compute lower-triangular square root of M via LLT, falling back to LDLT
// (with negative D clamped to zero). Throws CovarianceNotPsd if both fail.
template <int N>
Eigen::Matrix<double, N, N> safe_cholesky(const Eigen::Matrix<double, N, N> &M) {
  using Mat = Eigen::Matrix<double, N, N>;

  Eigen::LLT<Mat> llt(M);
  if (llt.info() == Eigen::Success) {
    return llt.matrixL();
  }

  const Mat sym = 0.5 * (M + M.transpose());
  Eigen::SelfAdjointEigenSolver<Mat> eig(sym);
  if (eig.info() != Eigen::Success) {
    throw CovarianceNotPsd("safe_cholesky: matrix not positive semi-definite");
  }
  const double max_abs_eigen = eig.eigenvalues().cwiseAbs().maxCoeff();
  const double tol = 1e-10 * std::max(1.0, max_abs_eigen);
  if (eig.eigenvalues().minCoeff() < -tol) {
    throw CovarianceNotPsd("safe_cholesky: matrix has negative eigenvalues");
  }
  const auto sqrt_eigenvalues = eig.eigenvalues().array().max(0.0).sqrt().matrix();
  return eig.eigenvectors() * sqrt_eigenvalues.asDiagonal();
}

// Generate 2N+1 Merwe scaled sigma points from mean mu and covariance sigma.
// Columns are: chi_0 = mu, chi_i = mu + sqrt((N+lambda) sigma)_i,
//              chi_{N+i} = mu - sqrt((N+lambda) sigma)_i.
template <int N>
Eigen::Matrix<double, N, 2 * N + 1> generate_sigma_points(const Eigen::Matrix<double, N, 1> &mu,
                                                          const Eigen::Matrix<double, N, N> &sigma,
                                                          double lambda) {
  using Mat = Eigen::Matrix<double, N, N>;
  using Points = Eigen::Matrix<double, N, 2 * N + 1>;

  const Mat scaled = (static_cast<double>(N) + lambda) * sigma;
  const Mat L = safe_cholesky<N>(scaled);

  Points points;
  points.col(0) = mu;
  for (int i = 0; i < N; ++i) {
    points.col(i + 1) = mu + L.col(i);
    points.col(i + 1 + N) = mu - L.col(i);
  }
  return points;
}

}  // namespace immtrack::detail
