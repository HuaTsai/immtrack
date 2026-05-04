#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <immtrack/detail/sigma_points.hpp>
#include <immtrack/errors.hpp>

#include "eigen_matchers.hpp"

using immtrack::test::IsApprox;
using immtrack::detail::generate_sigma_points;
using immtrack::detail::UnscentedWeights;

TEMPLATE_TEST_CASE_SIG("Sigma points reconstruct mean and cov", "[sigma]",
                       ((int N), N), 3, 5, 7, 10) {
    using Vec = Eigen::Matrix<double, N, 1>;
    using Mat = Eigen::Matrix<double, N, N>;

    Vec mu = Vec::LinSpaced(1.0, static_cast<double>(N));
    Mat sigma = Mat::Identity() * 0.5;

    constexpr double alpha = 1e-3;
    constexpr double beta = 2.0;
    constexpr double kappa = 0.0;
    const auto weights = UnscentedWeights<N>::make(alpha, beta, kappa);

    const auto points = generate_sigma_points<N>(mu, sigma, weights.lambda);

    // Reconstruct mean: must equal mu (linear combination of symmetric points).
    Vec reconstructed_mean = Vec::Zero();
    for (int i = 0; i < 2 * N + 1; ++i) {
        reconstructed_mean += weights.mean_weights(i) * points.col(i);
    }
    // Tolerance reflects floating-point conditioning of Merwe weights
    // with alpha=1e-3: W_m_0 ~= -lambda/(N+lambda) is large negative,
    // remaining weights cancel it by sum ~= 1, costing ~6 digits of
    // precision on values of order |mu|.
    REQUIRE_THAT(reconstructed_mean, IsApprox(mu, 1e-8));

    // Reconstruct covariance.
    Mat reconstructed_cov = Mat::Zero();
    for (int i = 0; i < 2 * N + 1; ++i) {
        const Vec d = points.col(i) - mu;
        reconstructed_cov += weights.cov_weights(i) * d * d.transpose();
    }
    REQUIRE_THAT(reconstructed_cov, IsApprox(sigma, 1e-8));
}

TEST_CASE("Sigma point 0 equals mean", "[sigma]") {
    constexpr int N = 4;
    using Vec = Eigen::Matrix<double, N, 1>;
    using Mat = Eigen::Matrix<double, N, N>;

    Vec mu;
    mu << 1.0, -2.0, 3.5, 0.0;
    Mat sigma = Mat::Identity();

    const auto weights = UnscentedWeights<N>::make(1e-3, 2.0, 0.0);
    const auto points = generate_sigma_points<N>(mu, sigma, weights.lambda);

    REQUIRE_THAT(points.col(0), IsApprox(mu, 1e-12));
}

TEST_CASE("Sigma points are symmetric around mean", "[sigma]") {
    constexpr int N = 3;
    using Vec = Eigen::Matrix<double, N, 1>;
    using Mat = Eigen::Matrix<double, N, N>;

    Vec mu = Vec::Zero();
    Mat sigma = Mat::Identity() * 2.0;
    const auto weights = UnscentedWeights<N>::make(1e-3, 2.0, 0.0);
    const auto points = generate_sigma_points<N>(mu, sigma, weights.lambda);

    for (int i = 0; i < N; ++i) {
        const Vec sum = points.col(i + 1) + points.col(i + 1 + N);
        REQUIRE_THAT(sum, IsApprox(2.0 * mu, 1e-12));
    }
}

TEST_CASE("generate_sigma_points throws on negative-definite cov", "[sigma]") {
    constexpr int N = 3;
    using Vec = Eigen::Matrix<double, N, 1>;
    using Mat = Eigen::Matrix<double, N, N>;

    Vec mu = Vec::Zero();
    Mat sigma = -Mat::Identity();
    const auto weights = UnscentedWeights<N>::make(1e-3, 2.0, 0.0);

    REQUIRE_THROWS_AS(
        generate_sigma_points<N>(mu, sigma, weights.lambda),
        immtrack::CovarianceNotPsd);
}

TEST_CASE("UnscentedWeights row sums equal 1 for mean weights", "[sigma]") {
    const auto weights = UnscentedWeights<7>::make(1e-3, 2.0, 0.0);
    REQUIRE_THAT(weights.mean_weights.sum(),
                 Catch::Matchers::WithinAbs(1.0, 1e-9));
}
