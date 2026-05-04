#pragma once

#include <stdexcept>

namespace immtrack {

class CovarianceNotPsd : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class InvalidArgument : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

class NumericalError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace immtrack
