#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace hdos {

enum class LinearRegressionSolver {
    cholesky,
    svd
};

struct LinearRegressionOptions {
    bool fit_intercept = true;
    double l2_penalty = 0.0;

    LinearRegressionSolver solver =
        LinearRegressionSolver::cholesky;

    // Relative singular-value cutoff.
    // Zero means choose the tolerance automatically.
    double svd_rcond = 0.0;
};

class LinearRegression {
public:
    LinearRegression(
        std::size_t n_features,
        LinearRegressionOptions options = {}
    );

    void fit(
        std::span<const double> X,
        std::span<const double> y
    );

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
    std::size_t parameter_count() const noexcept;

    void make_design_row(
        std::span<const double> x,
        std::span<double> row
    ) const;

    void update_cholesky_state(
        std::span<const double> row,
        double y
    );

    void update_qr_state(
        std::span<const double> row,
        double y
    );

    void solve_if_needed();
    void solve_cholesky();
    void solve_svd();

    void unpack_solution(
        std::span<const double> solution
    );

    std::size_t n_features_;
    std::size_t n_observations_ = 0;

    LinearRegressionOptions options_;

    // Cholesky backend state.
    std::vector<double> root_variance_;
    std::vector<double> normal_equation_rhs_;

    // SVD backend state:
    //
    //     X = Q R,
    //     qr_rhs_ = Q^T y.
    //
    // Only the part relevant to the least-squares solution is retained.
    std::vector<double> qr_factor_;
    std::vector<double> qr_rhs_;

    double sum_of_squares_ = 0.0;

    std::vector<double> coefficients_;
    double intercept_ = 0.0;

    bool solution_is_current_ = false;
};

} 