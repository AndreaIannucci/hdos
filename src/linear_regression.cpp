#include "hdos/linear_regression.hpp"
#include "detail/cholesky.hpp"

namespace hdos {
    LinearRegression::LinearRegression(
        std::size_t n_features,
        LinearRegressionOptions options): n_features_(n_features), options_(options){}
    

    void LinearRegression::fit(
    std::span<const double> X,
    std::span<const double> y
)
{
    const std::size_t n_samples = y.size();

    if (n_samples == 0) {
        throw std::invalid_argument("The response cannot be empty");
    }

    if (X.size() != n_samples * n_features_) {
        throw std::invalid_argument("Incompatible dimensions");
    }

    const std::size_t n_coeff =
        n_features_ + (options_.fit_intercept ? 1 : 0);

    // Only the lower triangle of the Gram matrix is populated.
    std::vector<double> gram_matrix(n_coeff * n_coeff, 0.0);
    std::vector<double> feature_sums(n_features_, 0.0);
    normal_equation_rhs_.assign(n_coeff, 0.0);

    double response_sum = 0.0;
    sum_of_squares_ = 0.0;

    for (std::size_t i = 0; i < n_samples; ++i) {
        response_sum += y[i];
        sum_of_squares_ += y[i] * y[i];
    }

    for (std::size_t k = 0; k < n_features_; ++k) {
        const std::size_t k_offset = k * n_samples;

        double diagonal = 0.0;
        double cross_product = 0.0;
        double feature_sum = 0.0;

        for (std::size_t i = 0; i < n_samples; ++i) {
            const double value = X[k_offset + i];

            diagonal += value * value;
            cross_product += value * y[i];
            feature_sum += value;
        }

        gram_matrix[k + k * n_coeff] = diagonal;
        normal_equation_rhs_[k] = cross_product;
        feature_sums[k] = feature_sum;

        // Complete column k below the diagonal.
        for (std::size_t j = k + 1; j < n_features_; ++j) {
            const std::size_t j_offset = j * n_samples;
            double cross = 0.0;

            for (std::size_t i = 0; i < n_samples; ++i) {
                cross += X[j_offset + i] * X[k_offset + i];
            }

            gram_matrix[j + k * n_coeff] = cross;
        }
    }

    if (options_.fit_intercept) {
        const std::size_t intercept = n_features_;

        for (std::size_t k = 0; k < n_features_; ++k) {
            gram_matrix[intercept + k * n_coeff] =
                feature_sums[k];
        }

        gram_matrix[intercept + intercept * n_coeff] =
            static_cast<double>(n_samples);

        normal_equation_rhs_[intercept] = response_sum;
    }

    root_variance_ = detail::cholesky_decomp(gram_matrix);
    n_observations_ = n_samples;
};
}