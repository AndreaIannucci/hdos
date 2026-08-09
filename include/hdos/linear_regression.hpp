#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace hdos {

/// Numerical backend used to solve the least-squares problem.
enum class LinearRegressionSolver {
    /// Solve the regularised normal equations using Cholesky
    /// factorisation. This is the default and fastest backend,
    /// but an unregularised rank-deficient system cannot be solved.
    cholesky,

    /// Maintain a compact QR representation and solve it using
    /// singular value decomposition. This backend supports
    /// rank-deficient and underdetermined systems.
    svd
};

/// Configuration options for LinearRegression.
struct LinearRegressionOptions {
    /// Whether to include an intercept in the model.
    bool fit_intercept = true;

    /// Non-negative L2 penalty applied to feature coefficients.
    /// The intercept is never penalised.
    double l2_penalty = 0.0;

    /// Numerical backend used to solve the regression problem.
    LinearRegressionSolver solver =
        LinearRegressionSolver::cholesky;

    /// Relative singular-value cutoff used by the SVD backend.
    ///
    /// Singular values satisfying
    ///
    ///     sigma <= svd_rcond * sigma_max
    ///
    /// are treated as zero. A value of zero selects an automatic
    /// tolerance based on machine precision and problem dimensions.
    double svd_rcond = 0.0;
};

/// Batch and online linear regression with optional ridge regularisation.
///
/// Input matrices use column-major storage. For a matrix with
/// n_samples rows and n_features columns, the element in row i and
/// column j is stored at
///
///     X[i + j * n_samples].
///
/// Calling fit() replaces the existing model state. Subsequent
/// observations can be incorporated using rk1_update() or
/// batch_update(). Coefficients and R-squared are computed lazily
/// when requested.
class LinearRegression {
public:
    /// Construct an unfitted regression model.
    ///
    /// @param n_features Number of predictor variables.
    /// @param options Model configuration.
    ///
    /// @throws std::invalid_argument if n_features is zero or an
    /// option is invalid.
    explicit LinearRegression(
        std::size_t n_features,
        LinearRegressionOptions options = {}
    );

    /// Fit the model from scratch.
    ///
    /// @param X Column-major n_samples by n_features design matrix.
    /// @param y Response vector containing n_samples values.
    ///
    /// @throws std::invalid_argument for empty, incompatible, or
    /// non-finite input.
    void fit(
        std::span<const double> X,
        std::span<const double> y
    );

    /// Incorporate one additional observation.
    ///
    /// @param x Observation containing n_features values.
    /// @param y Corresponding response.
    ///
    /// @throws std::invalid_argument for incompatible or non-finite
    /// input.
    void rk1_update(
        std::span<const double> x,
        double y
    );

    /// Incorporate a batch of additional observations.
    ///
    /// @param X Column-major n_samples by n_features design matrix.
    /// @param y Response vector containing n_samples values.
    ///
    /// @throws std::invalid_argument for empty, incompatible, or
    /// non-finite input.
    void batch_update(
        std::span<const double> X,
        std::span<const double> y
    );

    /// Predict the response for one observation.
    ///
    /// @throws std::logic_error if the model has not been fitted.
    /// @throws std::invalid_argument for incompatible or non-finite
    /// input.
    [[nodiscard]]
    double predict(std::span<const double> x);

    /// Return the fitted feature coefficients.
    ///
    /// The returned reference remains valid until the model is
    /// updated, fitted again, reset, or destroyed.
    ///
    /// @throws std::logic_error if the model has not been fitted.
    [[nodiscard]]
    const std::vector<double>& coefficients();

    /// Return the fitted intercept.
    ///
    /// Returns zero when fit_intercept is false.
    ///
    /// @throws std::logic_error if the model has not been fitted.
    [[nodiscard]]
    double intercept();

    /// Return the coefficient of determination.
    ///
    /// The total sum of squares is centred around the response mean,
    /// including when fit_intercept is false:
    ///
    ///     R^2 = 1 - RSS / TSS.
    ///
    /// The ridge penalty is not included in RSS. For a constant
    /// response, this method returns one for a numerically perfect
    /// fit and zero otherwise.
    ///
    /// @throws std::logic_error if the model has not been fitted.
    [[nodiscard]]
    double r_squared();

    /// Return the number of predictor variables.
    [[nodiscard]]
    std::size_t n_features() const noexcept;

    /// Return the number of observations currently represented.
    [[nodiscard]]
    std::size_t n_observations() const noexcept;

    /// Clear all fitted and accumulated state.
    void reset();

private:
    [[nodiscard]]
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

    /// Update the centred training sufficient statistics with one
    /// observation.
    ///
    /// n_observations_ must contain the number of observations before
    /// the new observation is incorporated.
    void update_training_statistics(
        std::span<const double> x,
        double y
    ) noexcept;

    void solve_if_needed();
    void solve_cholesky();
    void solve_svd();

    void unpack_solution(
        std::span<const double> solution
    );

    /// Compute the unpenalised training residual sum of squares using
    /// numerically stable centred sufficient statistics.
    [[nodiscard]]
    double residual_sum_of_squares() const;

    /// Refresh the cached coefficient of determination after solving.
    void update_r_squared();

    std::size_t n_features_;
    std::size_t n_observations_ = 0;

    LinearRegressionOptions options_;

    // Cholesky backend state.
    std::vector<double> root_variance_;
    std::vector<double> normal_equation_rhs_;

    // SVD backend state. These store the compact representation
    //
    //     X = Q R,        qr_rhs_ = Q^T y.
    std::vector<double> qr_factor_;
    std::vector<double> qr_rhs_;

    // Centred training sufficient statistics used to compute R-squared
    // without catastrophic cancellation for large feature or response
    // offsets.
    //
    // centered_feature_gram_ stores the lower triangle of
    //
    //     sum_i (x_i - mean(x)) (x_i - mean(x))^T.
    //
    // centered_feature_response_ stores
    //
    //     sum_i (x_i - mean(x)) (y_i - mean(y)).
    std::vector<double> feature_mean_;
    std::vector<double> centered_feature_gram_;
    std::vector<double> centered_feature_response_;

    // response_m2_ = sum_i (y_i - mean(y))^2.
    double response_mean_ = 0.0;
    double response_m2_ = 0.0;

    // Refreshed whenever the regression solution is refreshed.
    double r_squared_ = 0.0;

    std::vector<double> coefficients_;
    double intercept_ = 0.0;

    bool solution_is_current_ = false;
};

}  // namespace hdos