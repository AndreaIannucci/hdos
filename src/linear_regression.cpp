#include "hdos/linear_regression.hpp"

#include "detail/cholesky.hpp"
#include "detail/matrix_invert.hpp"
#include "detail/one_sided_jacobi.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hdos {

LinearRegression::LinearRegression(
    std::size_t n_features,
    LinearRegressionOptions options
)
    : n_features_(n_features),
      options_(options)
{
    if (n_features_ == 0) {
        throw std::invalid_argument(
            "The number of features cannot be zero"
        );
    }

    if (!std::isfinite(options_.l2_penalty) ||
        options_.l2_penalty < 0.0) {
        throw std::invalid_argument(
            "The L2 penalty must be finite and non-negative"
        );
    }

    if (!std::isfinite(options_.svd_rcond) ||
        options_.svd_rcond < 0.0) {
        throw std::invalid_argument(
            "The SVD relative tolerance must be finite and non-negative"
        );
    }

    reset();
}

std::size_t LinearRegression::parameter_count() const noexcept
{
    return n_features_ +
           static_cast<std::size_t>(options_.fit_intercept);
}

void LinearRegression::make_design_row(
    std::span<const double> x,
    std::span<double> row
) const
{
    if (x.size() != n_features_) {
        throw std::invalid_argument("Incompatible dimensions");
    }

    if (row.size() != parameter_count()) {
        throw std::invalid_argument(
            "Invalid design-row dimension"
        );
    }

    for (std::size_t j = 0; j < n_features_; ++j) {
        row[j] = x[j];
    }

    if (options_.fit_intercept) {
        row[n_features_] = 1.0;
    }
}

void LinearRegression::fit(
    std::span<const double> X,
    std::span<const double> y
)
{
    const std::size_t n_samples = y.size();

    if (n_samples == 0) {
        throw std::invalid_argument(
            "The response cannot be empty"
        );
    }

    if (X.size() != n_samples * n_features_) {
        throw std::invalid_argument(
            "Incompatible dimensions"
        );
    }

    for (double value : X) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Input contains non-finite values"
            );
        }
    }

    for (double value : y) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Response contains non-finite values"
            );
        }
    }

    reset();

    /*
     * For the SVD backend, accumulate the compact QR
     * representation one observation at a time.
     */
    if (options_.solver == LinearRegressionSolver::svd) {
        batch_update(X, y);
        return;
    }

    /*
     * Cholesky backend: directly construct the lower
     * triangle of X^T X and X^T y.
     */
    const std::size_t n_coeff = parameter_count();

    std::vector<double> gram_matrix(
        n_coeff * n_coeff,
        0.0
    );

    std::vector<double> feature_sums(
        n_features_,
        0.0
    );

    std::vector<double> normal_equation_rhs(
        n_coeff,
        0.0
    );

    /*
     * Centred sufficient statistics used for stable R-squared
     * computation.
     */
    std::vector<double> new_feature_mean(
        n_features_,
        0.0
    );

    std::vector<double> new_centered_feature_gram(
        n_features_ * n_features_,
        0.0
    );

    std::vector<double> new_centered_feature_response(
        n_features_,
        0.0
    );

    double response_sum = 0.0;
    double new_response_mean = 0.0;
    double new_response_m2 = 0.0;

    /*
     * Compute the centred feature and response statistics using
     * the multivariate Welford recurrence.
     */
    for (std::size_t i = 0; i < n_samples; ++i) {
        const double response = y[i];

        const double old_count =
            static_cast<double>(i);

        const double new_count =
            static_cast<double>(i + 1);

        const double merge_weight =
            old_count / new_count;

        const double response_delta =
            response - new_response_mean;

        response_sum += response;

        for (
            std::size_t column = 0;
            column < n_features_;
            ++column
        ) {
            const double column_delta =
                X[i + column * n_samples] -
                new_feature_mean[column];

            for (
                std::size_t row = column;
                row < n_features_;
                ++row
            ) {
                const double row_delta =
                    X[i + row * n_samples] -
                    new_feature_mean[row];

                new_centered_feature_gram[
                    row + column * n_features_
                ] +=
                    merge_weight *
                    row_delta *
                    column_delta;
            }

            new_centered_feature_response[column] +=
                merge_weight *
                column_delta *
                response_delta;
        }

        for (std::size_t j = 0; j < n_features_; ++j) {
            const double feature_delta =
                X[i + j * n_samples] -
                new_feature_mean[j];

            new_feature_mean[j] +=
                feature_delta / new_count;
        }

        new_response_mean +=
            response_delta / new_count;

        new_response_m2 +=
            merge_weight *
            response_delta *
            response_delta;
    }

    /*
     * Construct X^T X and X^T y.
     */
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
        normal_equation_rhs[k] = cross_product;
        feature_sums[k] = feature_sum;

        for (
            std::size_t j = k + 1;
            j < n_features_;
            ++j
        ) {
            const std::size_t j_offset =
                j * n_samples;

            double cross = 0.0;

            for (std::size_t i = 0; i < n_samples; ++i) {
                cross +=
                    X[j_offset + i] *
                    X[k_offset + i];
            }

            // Row j, column k in column-major storage.
            gram_matrix[j + k * n_coeff] = cross;
        }
    }

    if (options_.fit_intercept) {
        const std::size_t intercept = n_features_;

        for (std::size_t k = 0; k < n_features_; ++k) {
            gram_matrix[
                intercept + k * n_coeff
            ] = feature_sums[k];
        }

        gram_matrix[
            intercept + intercept * n_coeff
        ] = static_cast<double>(n_samples);

        normal_equation_rhs[intercept] =
            response_sum;
    }

    // Do not regularise the intercept.
    for (std::size_t j = 0; j < n_features_; ++j) {
        gram_matrix[j + j * n_coeff] +=
            options_.l2_penalty;
    }

    root_variance_ =
        detail::cholesky_decomp(gram_matrix);

    normal_equation_rhs_ =
        std::move(normal_equation_rhs);

    feature_mean_ =
        std::move(new_feature_mean);

    centered_feature_gram_ =
        std::move(new_centered_feature_gram);

    centered_feature_response_ =
        std::move(new_centered_feature_response);

    response_mean_ = new_response_mean;
    response_m2_ = new_response_m2;

    n_observations_ = n_samples;
    solution_is_current_ = false;
}

void LinearRegression::update_cholesky_state(
    std::span<const double> row,
    double y
)
{
    if (row.size() != parameter_count()) {
        throw std::invalid_argument(
            "Invalid design-row dimension"
        );
    }

    /*
     * rk1_cholesky expects only the original features.
     * It adds the intercept coordinate internally when
     * fit_intercept is true.
     */
    detail::rk1_cholesky(
        root_variance_,
        row.first(n_features_),
        1.0,
        options_.fit_intercept
    );

    for (std::size_t j = 0; j < parameter_count(); ++j) {
        normal_equation_rhs_[j] += row[j] * y;
    }
}

void LinearRegression::update_qr_state(
    std::span<const double> row,
    double y
)
{
    const std::size_t dimension = parameter_count();

    if (row.size() != dimension) {
        throw std::invalid_argument(
            "Invalid design-row dimension"
        );
    }

    /*
     * Triangularise
     *
     *      [ R ]
     *      [a^T]
     *
     * with Givens rotations. Apply the same rotations to
     *
     *      [ Q^T y ]
     *      [   y   ].
     */
    std::vector<double> work_row(
        row.begin(),
        row.end()
    );

    double work_response = y;

    for (std::size_t j = 0; j < dimension; ++j) {
        const double upper =
            qr_factor_[j + j * dimension];

        const double lower =
            work_row[j];

        if (lower == 0.0) {
            continue;
        }

        const double radius =
            std::hypot(upper, lower);

        if (radius == 0.0) {
            continue;
        }

        const double cosine =
            upper / radius;

        const double sine =
            lower / radius;

        for (
            std::size_t column = j;
            column < dimension;
            ++column
        ) {
            const std::size_t index =
                j + column * dimension;

            const double top =
                qr_factor_[index];

            const double bottom =
                work_row[column];

            qr_factor_[index] =
                cosine * top +
                sine * bottom;

            work_row[column] =
                -sine * top +
                cosine * bottom;
        }

        work_row[j] = 0.0;

        const double top_response =
            qr_rhs_[j];

        qr_rhs_[j] =
            cosine * top_response +
            sine * work_response;

        work_response =
            -sine * top_response +
            cosine * work_response;
    }

    /*
     * work_response is the new component of the residual.
     * It is irrelevant to the least-squares minimiser and
     * therefore need not be stored.
     */
}

void LinearRegression::rk1_update(
    std::span<const double> x,
    double y
)
{
    if (x.size() != n_features_) {
        throw std::invalid_argument(
            "Incompatible dimensions"
        );
    }

    for (double value : x) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Input contains non-finite values"
            );
        }
    }

    if (!std::isfinite(y)) {
        throw std::invalid_argument(
            "Response contains a non-finite value"
        );
    }

    std::vector<double> row(parameter_count());
    make_design_row(x, row);

    if (
        options_.solver ==
        LinearRegressionSolver::cholesky
    ) {
        update_cholesky_state(row, y);
    } else {
        update_qr_state(row, y);
    }

    /*
     * Update the centred training statistics only after the
     * matrix update has succeeded. At this point n_observations_
     * still contains the previous observation count.
     */
    update_training_statistics(x, y);

    ++n_observations_;
    solution_is_current_ = false;
}

void LinearRegression::update_training_statistics(
    std::span<const double> x,
    double y
) noexcept
{
    const double old_count =
        static_cast<double>(n_observations_);

    const double new_count =
        static_cast<double>(n_observations_ + 1);

    const double merge_weight =
        old_count / new_count;

    const double response_delta =
        y - response_mean_;

    for (
        std::size_t column = 0;
        column < n_features_;
        ++column
    ) {
        const double column_delta =
            x[column] - feature_mean_[column];

        for (
            std::size_t row = column;
            row < n_features_;
            ++row
        ) {
            const double row_delta =
                x[row] - feature_mean_[row];

            centered_feature_gram_[
                row + column * n_features_
            ] +=
                merge_weight *
                row_delta *
                column_delta;
        }

        centered_feature_response_[column] +=
            merge_weight *
            column_delta *
            response_delta;
    }

    for (std::size_t j = 0; j < n_features_; ++j) {
        const double feature_delta =
            x[j] - feature_mean_[j];

        feature_mean_[j] +=
            feature_delta / new_count;
    }

    response_mean_ +=
        response_delta / new_count;

    response_m2_ +=
        merge_weight *
        response_delta *
        response_delta;
}

void LinearRegression::batch_update(
    std::span<const double> X,
    std::span<const double> y
)
{
    const std::size_t n_samples = y.size();

    if (n_samples == 0) {
        throw std::invalid_argument(
            "The response cannot be empty"
        );
    }

    if (X.size() != n_samples * n_features_) {
        throw std::invalid_argument(
            "Incompatible dimensions"
        );
    }

    std::vector<double> observation(n_features_);

    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t j = 0; j < n_features_; ++j) {
            // Row i, column j in column-major storage.
            observation[j] =
                X[i + j * n_samples];
        }

        rk1_update(observation, y[i]);
    }
}

void LinearRegression::unpack_solution(
    std::span<const double> solution
)
{
    if (solution.size() != parameter_count()) {
        throw std::logic_error(
            "Invalid linear-system solution dimension"
        );
    }

    const auto coefficient_solution =
        solution.first(n_features_);

    coefficients_.assign(
        coefficient_solution.begin(),
        coefficient_solution.end()
    );

    if (options_.fit_intercept) {
        intercept_ = solution[n_features_];
    } else {
        intercept_ = 0.0;
    }
}

void LinearRegression::solve_cholesky()
{
    const std::vector<double> solution =
        detail::solve_pos_definite_cholesky(
            root_variance_,
            normal_equation_rhs_
        );

    unpack_solution(solution);
}

void LinearRegression::solve_svd()
{
    const std::size_t dimension = parameter_count();

    /*
     * For ridge regression, solve
     *
     *      [ R             ] beta ≈ [Q^T y]
     *      [sqrt(lambda) D ]        [  0  ]
     *
     * where D has ones on the feature coordinates and
     * zero on the intercept coordinate.
     */
    const std::size_t penalty_rows =
        options_.l2_penalty > 0.0
            ? n_features_
            : 0;

    const std::size_t system_rows =
        dimension + penalty_rows;

    std::vector<double> system(
        system_rows * dimension,
        0.0
    );

    std::vector<double> rhs(
        system_rows,
        0.0
    );

    for (
        std::size_t column = 0;
        column < dimension;
        ++column
    ) {
        for (
            std::size_t row = 0;
            row < dimension;
            ++row
        ) {
            system[row + column * system_rows] =
                qr_factor_[
                    row + column * dimension
                ];
        }
    }

    for (std::size_t row = 0; row < dimension; ++row) {
        rhs[row] = qr_rhs_[row];
    }

    if (options_.l2_penalty > 0.0) {
        const double ridge_scale =
            std::sqrt(options_.l2_penalty);

        for (std::size_t j = 0; j < n_features_; ++j) {
            system[
                (dimension + j) +
                j * system_rows
            ] = ridge_scale;
        }
    }

    const auto decomposition =
        detail::jacobi_svd(
            std::span<const double>(system),
            system_rows,
            dimension
        );

    double largest_singular_value = 0.0;

    for (
        double singular_value :
        decomposition.singular_values
    ) {
        largest_singular_value =
            std::max(
                largest_singular_value,
                singular_value
            );
    }

    double relative_tolerance =
        options_.svd_rcond;

    if (relative_tolerance == 0.0) {
        const std::size_t scale_dimension =
            std::max(
                n_observations_,
                dimension
            );

        relative_tolerance =
            std::numeric_limits<double>::epsilon() *
            static_cast<double>(scale_dimension);
    }

    const double cutoff =
        relative_tolerance *
        largest_singular_value;

    // Compute U^T rhs.
    std::vector<double> projected_rhs(
        dimension,
        0.0
    );

    for (std::size_t j = 0; j < dimension; ++j) {
        double value = 0.0;

        for (
            std::size_t i = 0;
            i < system_rows;
            ++i
        ) {
            value +=
                decomposition.U[
                    i + j * system_rows
                ] * rhs[i];
        }

        if (
            decomposition.singular_values[j] >
            cutoff
        ) {
            projected_rhs[j] =
                value /
                decomposition.singular_values[j];
        }
    }

    // Compute V Sigma^+ U^T rhs.
    std::vector<double> solution(
        dimension,
        0.0
    );

    for (std::size_t i = 0; i < dimension; ++i) {
        for (std::size_t j = 0; j < dimension; ++j) {
            solution[i] +=
                decomposition.V[
                    i + j * dimension
                ] * projected_rhs[j];
        }
    }

    unpack_solution(solution);
}

double LinearRegression::residual_sum_of_squares() const
{
    double residual_sum =
        response_m2_;

    /*
     * Centred residual contribution:
     *
     *     sum_i ((y_i - mean(y))
     *            - beta^T (x_i - mean(x)))^2.
     */
    for (std::size_t j = 0; j < n_features_; ++j) {
        residual_sum -=
            2.0 *
            coefficients_[j] *
            centered_feature_response_[j];
    }

    /*
     * Compute beta^T C_xx beta using the stored lower
     * triangle of C_xx.
     */
    for (
        std::size_t column = 0;
        column < n_features_;
        ++column
    ) {
        residual_sum +=
            coefficients_[column] *
            coefficients_[column] *
            centered_feature_gram_[
                column + column * n_features_
            ];

        for (
            std::size_t row = column + 1;
            row < n_features_;
            ++row
        ) {
            residual_sum +=
                2.0 *
                coefficients_[row] *
                coefficients_[column] *
                centered_feature_gram_[
                    row + column * n_features_
                ];
        }
    }

    /*
     * Add the contribution from the mean residual. This is
     * generally non-zero when the model has no intercept.
     */
    double mean_prediction =
        intercept_;

    for (std::size_t j = 0; j < n_features_; ++j) {
        mean_prediction +=
            feature_mean_[j] *
            coefficients_[j];
    }

    const double mean_residual =
        response_mean_ -
        mean_prediction;

    residual_sum +=
        static_cast<double>(n_observations_) *
        mean_residual *
        mean_residual;

    /*
     * Roundoff can produce a very small negative value even
     * though RSS is mathematically non-negative.
     */
    return std::max(0.0, residual_sum);
}

void LinearRegression::update_r_squared()
{
    const double rss =
        residual_sum_of_squares();

    /*
     * A response is constant precisely when its centred sum
     * of squares is zero.
     */
    if (response_m2_ <= 0.0) {
        const double response_scale =
            std::max(
                1.0,
                std::abs(response_mean_)
            );

        const double roundoff =
            std::numeric_limits<double>::epsilon() *
            response_scale;

        const double tolerance =
            100.0 *
            static_cast<double>(n_observations_) *
            roundoff *
            roundoff;

        r_squared_ =
            rss <= tolerance ? 1.0 : 0.0;

        return;
    }

    r_squared_ =
        1.0 - rss / response_m2_;
}

void LinearRegression::solve_if_needed()
{
    if (solution_is_current_) {
        return;
    }

    if (n_observations_ == 0) {
        throw std::logic_error(
            "The model has not been fitted"
        );
    }

    if (
        options_.solver ==
        LinearRegressionSolver::cholesky
    ) {
        solve_cholesky();
    } else {
        solve_svd();
    }

    update_r_squared();
    solution_is_current_ = true;
}

double LinearRegression::predict(
    std::span<const double> x
)
{
    if (x.size() != n_features_) {
        throw std::invalid_argument(
            "Incompatible dimensions"
        );
    }

    for (double value : x) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Input contains non-finite values"
            );
        }
    }

    solve_if_needed();

    double output =
        intercept_;

    for (std::size_t j = 0; j < n_features_; ++j) {
        output +=
            x[j] *
            coefficients_[j];
    }

    return output;
}

const std::vector<double>&
LinearRegression::coefficients()
{
    solve_if_needed();
    return coefficients_;
}

double LinearRegression::intercept()
{
    solve_if_needed();
    return intercept_;
}

double LinearRegression::r_squared()
{
    solve_if_needed();
    return r_squared_;
}

std::size_t
LinearRegression::n_features() const noexcept
{
    return n_features_;
}

std::size_t
LinearRegression::n_observations() const noexcept
{
    return n_observations_;
}

void LinearRegression::reset()
{
    const std::size_t dimension =
        parameter_count();

    n_observations_ = 0;

    feature_mean_.assign(
        n_features_,
        0.0
    );

    centered_feature_gram_.assign(
        n_features_ * n_features_,
        0.0
    );

    centered_feature_response_.assign(
        n_features_,
        0.0
    );

    response_mean_ = 0.0;
    response_m2_ = 0.0;
    r_squared_ = 0.0;

    root_variance_.assign(
        dimension * dimension,
        0.0
    );

    normal_equation_rhs_.assign(
        dimension,
        0.0
    );

    qr_factor_.assign(
        dimension * dimension,
        0.0
    );

    qr_rhs_.assign(
        dimension,
        0.0
    );

    /*
     * The Cholesky factor initially represents the ridge
     * matrix. The intercept remains unregularised.
     */
    if (options_.l2_penalty > 0.0) {
        const double ridge_scale =
            std::sqrt(options_.l2_penalty);

        for (std::size_t j = 0; j < n_features_; ++j) {
            root_variance_[
                j + j * dimension
            ] = ridge_scale;
        }
    }

    coefficients_.assign(
        n_features_,
        0.0
    );

    intercept_ = 0.0;
    solution_is_current_ = false;
}

} 