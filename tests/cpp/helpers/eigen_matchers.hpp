#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <sstream>
#include <string>

namespace immtrack::test {

// Approximate-equality matcher for any Eigen dense matrix.
class IsApprox : public Catch::Matchers::MatcherGenericBase {
 public:
  IsApprox(Eigen::MatrixXd expected, double tol) : expected_(std::move(expected)), tol_(tol) {}

  template <class Derived>
  bool match(const Eigen::MatrixBase<Derived> &actual) const {
    if (actual.rows() != expected_.rows() || actual.cols() != expected_.cols()) {
      return false;
    }
    return (actual - expected_).norm() < tol_;
  }

  std::string describe() const override {
    std::ostringstream oss;
    oss << "is approximately equal to expected matrix (tol = " << tol_ << ")";
    return oss.str();
  }

 private:
  Eigen::MatrixXd expected_;
  double tol_;
};

// Positive-semi-definite matcher: all eigenvalues >= -tol.
class IsPsd : public Catch::Matchers::MatcherGenericBase {
 public:
  explicit IsPsd(double tol = 1e-10) : tol_(tol) {}

  template <class Derived>
  bool match(const Eigen::MatrixBase<Derived> &actual) const {
    if (actual.rows() != actual.cols()) {
      return false;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(actual);
    if (solver.info() != Eigen::Success) {
      return false;
    }
    return solver.eigenvalues().minCoeff() >= -tol_;
  }

  std::string describe() const override {
    std::ostringstream oss;
    oss << "is positive semi-definite (tol = " << tol_ << ")";
    return oss.str();
  }

 private:
  double tol_;
};

}  // namespace immtrack::test
