#pragma once

#include <vector>
#include <cstddef>
#include <span>
#include <stdexcept>


namespace hdos{
struct LinearRegressionOptions{
    bool fit_intercept = true;
    double l2_penalty = 0.0;
};

class LinearRegression{
    public:
    
    LinearRegression(
    std::size_t n_features,
    LinearRegressionOptions options = {});

    // Modify only the sufficient statistics
    // Clear past estimates
    void fit(
        std::span<const double> X,
        std::span<const double> y
    );

    // Create design matrix, create covariance matrix and X^t y

    void rk1_update(
        std::span<const double> x,
        double y
    );

    void batch_update(
        std::span<const double> X,
        std::span<const double> y
    );

    double predict(std::span<const double> x);
    const std::vector<double>& coefficients();
    double intercept();

    std::size_t n_features() const noexcept;
    std::size_t n_observations() const noexcept;

    void reset();

    private:

    // Lazy solving
    void solve_if_needed();

    std::size_t n_features_;
    std::size_t n_observations_ = 0;

    LinearRegressionOptions options_;

    std::vector<double> root_variance_;
    std::vector<double> normal_equation_rhs_;
    double sum_of_squares_ = 0.0;

    std::vector<double> coefficients_;
    double intercept_ = 0.0;

    bool solution_is_current_ = false;
};
}