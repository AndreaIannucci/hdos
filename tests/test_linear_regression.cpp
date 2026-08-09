#include <gtest/gtest.h>

#include "hdos/linear_regression.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using hdos::LinearRegression;
using hdos::LinearRegressionOptions;
using hdos::LinearRegressionSolver;

constexpr double tolerance = 1e-9;

/*
 * Convert a matrix given as rows into the column-major
 * representation expected by LinearRegression.
 */
std::vector<double> to_column_major(
    const std::vector<std::vector<double>>& rows
)
{
    if (rows.empty()) {
        return {};
    }

    const std::size_t n_rows = rows.size();
    const std::size_t n_columns = rows.front().size();

    for (const auto& row : rows) {
        if (row.size() != n_columns) {
            throw std::invalid_argument(
                "Rows have incompatible dimensions"
            );
        }
    }

    std::vector<double> output(
        n_rows * n_columns,
        0.0
    );

    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < n_columns; ++j) {
            output[i + j * n_rows] = rows[i][j];
        }
    }

    return output;
}

LinearRegressionOptions make_options(
    LinearRegressionSolver solver,
    bool fit_intercept = true,
    double l2_penalty = 0.0,
    double svd_rcond = 0.0
)
{
    LinearRegressionOptions options;

    options.fit_intercept = fit_intercept;
    options.l2_penalty = l2_penalty;
    options.solver = solver;
    options.svd_rcond = svd_rcond;

    return options;
}

void expect_coefficients_near(
    LinearRegression& model,
    const std::vector<double>& expected,
    double expected_intercept,
    double error = tolerance
)
{
    const auto& coefficients = model.coefficients();

    ASSERT_EQ(coefficients.size(), expected.size());

    for (std::size_t j = 0; j < expected.size(); ++j) {
        EXPECT_NEAR(
            coefficients[j],
            expected[j],
            error
        ) << "Coefficient " << j << " differs";
    }

    EXPECT_NEAR(
        model.intercept(),
        expected_intercept,
        error
    );
}

class LinearRegressionSolverTest
    : public ::testing::TestWithParam<
          LinearRegressionSolver
      > {
};

INSTANTIATE_TEST_SUITE_P(
    AllSolvers,
    LinearRegressionSolverTest,
    ::testing::Values(
        LinearRegressionSolver::cholesky,
        LinearRegressionSolver::svd
    ),
    [](
        const ::testing::TestParamInfo<
            LinearRegressionSolver
        >& information
    ) {
        if (
            information.param ==
            LinearRegressionSolver::cholesky
        ) {
            return "Cholesky";
        }

        return "SVD";
    }
);

/*
 * Constructor and options validation.
 */

TEST(LinearRegressionConstructorTest, RejectsZeroFeatures)
{
    EXPECT_THROW(
        LinearRegression(0),
        std::invalid_argument
    );
}

TEST(LinearRegressionConstructorTest, RejectsNegativePenalty)
{
    LinearRegressionOptions options;
    options.l2_penalty = -1.0;

    EXPECT_THROW(
        LinearRegression(2, options),
        std::invalid_argument
    );
}

TEST(LinearRegressionConstructorTest, RejectsNanPenalty)
{
    LinearRegressionOptions options;
    options.l2_penalty =
        std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(
        LinearRegression(2, options),
        std::invalid_argument
    );
}

TEST(LinearRegressionConstructorTest, RejectsInfinitePenalty)
{
    LinearRegressionOptions options;
    options.l2_penalty =
        std::numeric_limits<double>::infinity();

    EXPECT_THROW(
        LinearRegression(2, options),
        std::invalid_argument
    );
}

TEST(LinearRegressionConstructorTest, RejectsNegativeSvdTolerance)
{
    LinearRegressionOptions options;
    options.svd_rcond = -1e-8;

    EXPECT_THROW(
        LinearRegression(2, options),
        std::invalid_argument
    );
}

TEST(LinearRegressionConstructorTest, RejectsNanSvdTolerance)
{
    LinearRegressionOptions options;
    options.svd_rcond =
        std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(
        LinearRegression(2, options),
        std::invalid_argument
    );
}

TEST(LinearRegressionConstructorTest, ExposesModelDimensions)
{
    LinearRegression model(4);

    EXPECT_EQ(model.n_features(), 4);
    EXPECT_EQ(model.n_observations(), 0);
}

/*
 * Basic exact regression tests.
 */

TEST_P(LinearRegressionSolverTest, FitsExactLineWithIntercept)
{
    const auto solver = GetParam();

    LinearRegression model(
        1,
        make_options(solver)
    );

    // y = 1 + 2x.
    const std::vector<double> X{
        0.0,
        1.0,
        2.0,
        3.0
    };

    const std::vector<double> y{
        1.0,
        3.0,
        5.0,
        7.0
    };

    model.fit(X, y);

    EXPECT_EQ(model.n_observations(), 4);

    expect_coefficients_near(
        model,
        {2.0},
        1.0
    );

    const std::array<double, 1> input{4.0};

    EXPECT_NEAR(
        model.predict(input),
        9.0,
        tolerance
    );
}

TEST_P(LinearRegressionSolverTest, FitsExactLineWithoutIntercept)
{
    const auto solver = GetParam();

    LinearRegression model(
        1,
        make_options(
            solver,
            false
        )
    );

    // y = 3x.
    const std::vector<double> X{
        1.0,
        2.0,
        3.0,
        4.0
    };

    const std::vector<double> y{
        3.0,
        6.0,
        9.0,
        12.0
    };

    model.fit(X, y);

    expect_coefficients_near(
        model,
        {3.0},
        0.0
    );

    const std::array<double, 1> input{5.0};

    EXPECT_NEAR(
        model.predict(input),
        15.0,
        tolerance
    );
}

TEST_P(LinearRegressionSolverTest, FitsMultipleFeaturesWithIntercept)
{
    const auto solver = GetParam();

    LinearRegression model(
        2,
        make_options(solver)
    );

    /*
     * y = 1 + 2x1 - 3x2.
     */
    const std::vector<double> X = to_column_major({
        {0.0,  0.0},
        {1.0,  0.0},
        {0.0,  1.0},
        {1.0,  1.0},
        {2.0, -1.0}
    });

    const std::vector<double> y{
         1.0,
         3.0,
        -2.0,
         0.0,
         8.0
    };

    model.fit(X, y);

    expect_coefficients_near(
        model,
        {2.0, -3.0},
        1.0
    );

    const std::array<double, 2> input{
        3.0,
        2.0
    };

    EXPECT_NEAR(
        model.predict(input),
        1.0,
        tolerance
    );
}

TEST_P(LinearRegressionSolverTest, FitsMultipleFeaturesWithoutIntercept)
{
    const auto solver = GetParam();

    LinearRegression model(
        2,
        make_options(
            solver,
            false
        )
    );

    /*
     * y = 2x1 - 3x2.
     */
    const std::vector<double> X = to_column_major({
        { 1.0, 0.0},
        { 0.0, 1.0},
        { 1.0, 1.0},
        {-1.0, 2.0}
    });

    const std::vector<double> y{
         2.0,
        -3.0,
        -1.0,
        -8.0
    };

    model.fit(X, y);

    expect_coefficients_near(
        model,
        {2.0, -3.0},
        0.0
    );
}

TEST_P(LinearRegressionSolverTest, UsesColumnMajorInput)
{
    const auto solver = GetParam();

    LinearRegression model(
        2,
        make_options(solver)
    );

    /*
     * Rows:
     *
     *     (1, 10)
     *     (2, 20)
     *     (4, 10)
     *     (5, 30)
     *
     * y = 4 + 3x1 - 2x2.
     */
    const std::vector<double> X{
        1.0, 2.0, 4.0, 5.0,
        10.0, 20.0, 10.0, 30.0
    };

    const std::vector<double> y{
        -13.0,
        -30.0,
         -4.0,
        -41.0
    };

    model.fit(X, y);

    expect_coefficients_near(
        model,
        {3.0, -2.0},
        4.0,
        1e-8
    );
}

/*
 * State replacement, reset and lazy solution tests.
 */

TEST_P(LinearRegressionSolverTest, FitReplacesPreviousState)
{
    const auto solver = GetParam();

    LinearRegression model(
        1,
        make_options(solver)
    );

    const std::vector<double> first_X{
        0.0,
        1.0,
        2.0
    };

    const std::vector<double> first_y{
        1.0,
        3.0,
        5.0
    };

    model.fit(first_X, first_y);

    expect_coefficients_near(
        model,
        {2.0},
        1.0
    );

    // y = -2 + 0.5x.
    const std::vector<double> second_X{
        -2.0,
        -1.0,
         1.0,
         3.0
    };

    const std::vector<double> second_y{
        -3.0,
        -2.5,
        -1.5,
        -0.5
    };

    model.fit(second_X, second_y);

    EXPECT_EQ(model.n_observations(), 4);

    expect_coefficients_near(
        model,
        {0.5},
        -2.0
    );
}

TEST_P(LinearRegressionSolverTest, ResetClearsModelState)
{
    const auto solver = GetParam();

    LinearRegression model(
        1,
        make_options(solver)
    );

    const std::vector<double> X{
        0.0,
        1.0,
        2.0
    };

    const std::vector<double> y{
        1.0,
        3.0,
        5.0
    };

    model.fit(X, y);

    EXPECT_EQ(model.n_observations(), 3);

    // Force the lazy solution to be computed.
    EXPECT_NEAR(
        model.coefficients()[0],
        2.0,
        tolerance
    );

    model.reset();

    EXPECT_EQ(model.n_observations(), 0);

    EXPECT_THROW(
        static_cast<void>(model.coefficients()),
        std::logic_error
    );

    EXPECT_THROW(
        static_cast<void>(model.intercept()),
        std::logic_error
    );

    const std::array<double, 1> input{1.0};

    EXPECT_THROW(
        static_cast<void>(model.predict(input)),
        std::logic_error
    );
}

TEST_P(LinearRegressionSolverTest, CanFitAgainAfterReset)
{
    const auto solver = GetParam();

    LinearRegression model(
        1,
        make_options(solver)
    );

    const std::vector<double> first_X{
        0.0,
        1.0,
        2.0
    };

    const std::vector<double> first_y{
        1.0,
        3.0,
        5.0
    };

    model.fit(first_X, first_y);
    model.reset();

    const std::vector<double> second_X{
        0.0,
        1.0,
        2.0
    };

    const std::vector<double> second_y{
        -1.0,
         2.0,
         5.0
    };

    model.fit(second_X, second_y);

    expect_coefficients_near(
        model,
        {3.0},
        -1.0
    );
}

TEST_P(LinearRegressionSolverTest, PredictTriggersLazySolution)
{
    const auto solver = GetParam();

    LinearRegression model(
        1,
        make_options(solver)
    );

    const std::vector<double> X{
        0.0,
        1.0,
        2.0
    };

    const std::vector<double> y{
        1.0,
        3.0,
        5.0
    };

    model.fit(X, y);

    const std::array<double, 1> input{3.0};

    EXPECT_NEAR(
        model.predict(input),
        7.0,
        tolerance
    );
}

/*
 * Online update tests.
 */

TEST_P(LinearRegressionSolverTest, RankOneUpdatesMatchCompleteRefit)
{
    const auto solver = GetParam();

    LinearRegression updated_model(
        2,
        make_options(solver)
    );

    LinearRegression reference_model(
        2,
        make_options(solver)
    );

    /*
     * The first three observations already provide a
     * full-rank system.
     */
    const std::vector<double> initial_X = to_column_major({
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0}
    });

    const std::vector<double> initial_y{
         1.0,
         3.0,
        -2.0
    };

    updated_model.fit(initial_X, initial_y);

    const std::array<double, 2> fourth_x{
        1.0,
        1.0
    };

    const std::array<double, 2> fifth_x{
         2.0,
        -1.0
    };

    updated_model.rk1_update(fourth_x, 0.0);
    updated_model.rk1_update(fifth_x, 8.0);

    const std::vector<double> complete_X = to_column_major({
        {0.0,  0.0},
        {1.0,  0.0},
        {0.0,  1.0},
        {1.0,  1.0},
        {2.0, -1.0}
    });

    const std::vector<double> complete_y{
         1.0,
         3.0,
        -2.0,
         0.0,
         8.0
    };

    reference_model.fit(
        complete_X,
        complete_y
    );

    EXPECT_EQ(
        updated_model.n_observations(),
        5
    );

    const auto& updated_coefficients =
        updated_model.coefficients();

    const auto& reference_coefficients =
        reference_model.coefficients();

    ASSERT_EQ(
        updated_coefficients.size(),
        reference_coefficients.size()
    );

    for (
        std::size_t j = 0;
        j < updated_coefficients.size();
        ++j
    ) {
        EXPECT_NEAR(
            updated_coefficients[j],
            reference_coefficients[j],
            tolerance
        );
    }

    EXPECT_NEAR(
        updated_model.intercept(),
        reference_model.intercept(),
        tolerance
    );
}

TEST_P(LinearRegressionSolverTest, BatchUpdateMatchesRankOneUpdates)
{
    const auto solver = GetParam();

    LinearRegression batch_model(
        2,
        make_options(solver)
    );

    LinearRegression sequential_model(
        2,
        make_options(solver)
    );

    const std::vector<double> initial_X = to_column_major({
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0}
    });

    const std::vector<double> initial_y{
         1.0,
         3.0,
        -2.0
    };

    batch_model.fit(initial_X, initial_y);
    sequential_model.fit(initial_X, initial_y);

    const std::vector<double> additional_X = to_column_major({
        {1.0,  1.0},
        {2.0, -1.0}
    });

    const std::vector<double> additional_y{
        0.0,
        8.0
    };

    batch_model.batch_update(
        additional_X,
        additional_y
    );

    const std::array<double, 2> first{
        1.0,
        1.0
    };

    const std::array<double, 2> second{
         2.0,
        -1.0
    };

    sequential_model.rk1_update(first, 0.0);
    sequential_model.rk1_update(second, 8.0);

    const auto& batch_coefficients =
        batch_model.coefficients();

    const auto& sequential_coefficients =
        sequential_model.coefficients();

    ASSERT_EQ(
        batch_coefficients.size(),
        sequential_coefficients.size()
    );

    for (
        std::size_t j = 0;
        j < batch_coefficients.size();
        ++j
    ) {
        EXPECT_NEAR(
            batch_coefficients[j],
            sequential_coefficients[j],
            tolerance
        );
    }

    EXPECT_NEAR(
        batch_model.intercept(),
        sequential_model.intercept(),
        tolerance
    );
}

TEST_P(LinearRegressionSolverTest, SolutionIsRefreshedAfterUpdate)
{
    const auto solver = GetParam();

    LinearRegression model(
        1,
        make_options(solver)
    );

    const std::vector<double> X{
        0.0,
        1.0
    };

    const std::vector<double> y{
        0.0,
        1.0
    };

    model.fit(X, y);

    expect_coefficients_near(
        model,
        {1.0},
        0.0
    );

    const std::array<double, 1> new_x{2.0};
    model.rk1_update(new_x, 4.0);

    /*
     * Least-squares fit through:
     *
     *     (0, 0), (1, 1), (2, 4)
     *
     * gives slope 2 and intercept -1/3.
     */
    expect_coefficients_near(
        model,
        {2.0},
        -1.0 / 3.0
    );
}

/*
 * Ridge regression tests.
 */

TEST_P(LinearRegressionSolverTest, RidgeWithoutInterceptHasKnownSolution)
{
    const auto solver = GetParam();

    /*
     * beta =
     *
     *     sum x_i y_i
     *     -----------
     *     sum x_i^2 + lambda
     *
     * Here:
     *
     *     sum x_i y_i = 10,
     *     sum x_i^2   = 5,
     *     lambda      = 5,
     *
     * so beta = 1.
     */
    LinearRegression model(
        1,
        make_options(
            solver,
            false,
            5.0
        )
    );

    const std::vector<double> X{
        1.0,
        2.0
    };

    const std::vector<double> y{
        2.0,
        4.0
    };

    model.fit(X, y);

    expect_coefficients_near(
        model,
        {1.0},
        0.0
    );
}

TEST_P(LinearRegressionSolverTest, RidgeDoesNotPenalizeIntercept)
{
    const auto solver = GetParam();

    /*
     * x is centred and y = 5 + 2x.
     *
     * With lambda = 3:
     *
     *     beta = 4 / (2 + 3) = 0.8.
     *
     * The intercept remains equal to mean(y) = 5.
     */
    LinearRegression model(
        1,
        make_options(
            solver,
            true,
            3.0
        )
    );

    const std::vector<double> X{
        -1.0,
         0.0,
         1.0
    };

    const std::vector<double> y{
        3.0,
        5.0,
        7.0
    };

    model.fit(X, y);

    expect_coefficients_near(
        model,
        {0.8},
        5.0
    );
}

TEST_P(
    LinearRegressionSolverTest,
    RidgeHandlesUnderdeterminedSystem
)
{
    const auto solver = GetParam();

    /*
     * Minimise
     *
     *     (beta_1 + 2 beta_2 - 5)^2
     *       + 5(beta_1^2 + beta_2^2).
     *
     * The solution is (0.5, 1.0).
     */
    LinearRegression model(
        2,
        make_options(
            solver,
            false,
            5.0
        )
    );

    // One row: (1, 2).
    const std::vector<double> X{
        1.0,
        2.0
    };

    const std::vector<double> y{
        5.0
    };

    model.fit(X, y);

    expect_coefficients_near(
        model,
        {0.5, 1.0},
        0.0
    );
}

TEST_P(
    LinearRegressionSolverTest,
    RidgeRankOneUpdatesMatchRefit
)
{
    const auto solver = GetParam();

    constexpr double penalty = 0.75;

    LinearRegression updated_model(
        2,
        make_options(
            solver,
            true,
            penalty
        )
    );

    LinearRegression reference_model(
        2,
        make_options(
            solver,
            true,
            penalty
        )
    );

    const std::vector<double> initial_X = to_column_major({
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0}
    });

    const std::vector<double> initial_y{
         1.0,
         3.0,
        -2.0
    };

    updated_model.fit(initial_X, initial_y);

    const std::array<double, 2> new_x{
        1.0,
        1.0
    };

    updated_model.rk1_update(new_x, 0.0);

    const std::vector<double> complete_X = to_column_major({
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    });

    const std::vector<double> complete_y{
         1.0,
         3.0,
        -2.0,
         0.0
    };

    reference_model.fit(
        complete_X,
        complete_y
    );

    const auto& updated_coefficients =
        updated_model.coefficients();

    const auto& reference_coefficients =
        reference_model.coefficients();

    ASSERT_EQ(
        updated_coefficients.size(),
        reference_coefficients.size()
    );

    for (
        std::size_t j = 0;
        j < updated_coefficients.size();
        ++j
    ) {
        EXPECT_NEAR(
            updated_coefficients[j],
            reference_coefficients[j],
            tolerance
        );
    }

    EXPECT_NEAR(
        updated_model.intercept(),
        reference_model.intercept(),
        tolerance
    );
}

/*
 * SVD-specific rank-deficient tests.
 */

TEST(LinearRegressionSvdTest, ComputesMinimumNormRankDeficientSolution)
{
    LinearRegression model(
        2,
        make_options(
            LinearRegressionSolver::svd
        )
    );

    /*
     * x2 = 2x1 and y = 3x1.
     *
     * We require
     *
     *     beta_1 + 2 beta_2 = 3.
     *
     * The minimum-norm solution is
     *
     *     beta = 3(1,2) / 5
     *          = (0.6, 1.2).
     */
    const std::vector<double> X = to_column_major({
        {-1.0, -2.0},
        { 0.0,  0.0},
        { 1.0,  2.0}
    });

    const std::vector<double> y{
        -3.0,
         0.0,
         3.0
    };

    model.fit(X, y);

    expect_coefficients_near(
        model,
        {0.6, 1.2},
        0.0,
        1e-8
    );
}

TEST(LinearRegressionSvdTest, ComputesMinimumNormUnderdeterminedSolution)
{
    LinearRegression model(
        2,
        make_options(
            LinearRegressionSolver::svd,
            false
        )
    );

    /*
     * One equation:
     *
     *     beta_1 + 2 beta_2 = 5.
     *
     * The minimum-norm solution is (1,2).
     */
    const std::vector<double> X{
        1.0,
        2.0
    };

    const std::vector<double> y{
        5.0
    };

    model.fit(X, y);

    expect_coefficients_near(
        model,
        {1.0, 2.0},
        0.0,
        1e-8
    );
}

TEST(LinearRegressionSvdTest, ReturnsZeroForZeroDesignMatrix)
{
    LinearRegression model(
        2,
        make_options(
            LinearRegressionSolver::svd,
            false
        )
    );

    const std::vector<double> X{
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    };

    const std::vector<double> y{
         1.0,
        -2.0,
         3.0
    };

    model.fit(X, y);

    expect_coefficients_near(
        model,
        {0.0, 0.0},
        0.0
    );
}

TEST(LinearRegressionSvdTest, AppliesExplicitRankCutoff)
{
    LinearRegression model(
        2,
        make_options(
            LinearRegressionSolver::svd,
            false,
            0.0,
            1e-8
        )
    );

    /*
     * Diagonal design with singular values 1 and 1e-10.
     * The second singular direction should be discarded.
     */
    const std::vector<double> X{
        1.0, 0.0,
        0.0, 1e-10
    };

    const std::vector<double> y{
        1.0,
        1e-10
    };

    model.fit(X, y);

    expect_coefficients_near(
        model,
        {1.0, 0.0},
        0.0,
        1e-8
    );
}

TEST(LinearRegressionSvdTest, AutomaticCutoffKeepsResolvableDirection)
{
    LinearRegression model(
        2,
        make_options(
            LinearRegressionSolver::svd,
            false
        )
    );

    const std::vector<double> X{
        1.0, 0.0,
        0.0, 1e-10
    };

    const std::vector<double> y{
        1.0,
        1e-10
    };

    model.fit(X, y);

    expect_coefficients_near(
        model,
        {1.0, 1.0},
        0.0,
        1e-7
    );
}

/*
 * Input validation tests.
 */

TEST_P(LinearRegressionSolverTest, FitRejectsEmptyResponse)
{
    const auto solver = GetParam();

    LinearRegression model(
        2,
        make_options(solver)
    );

    const std::vector<double> X;
    const std::vector<double> y;

    EXPECT_THROW(
        model.fit(X, y),
        std::invalid_argument
    );
}

TEST_P(LinearRegressionSolverTest, FitRejectsIncompatibleDimensions)
{
    const auto solver = GetParam();

    LinearRegression model(
        2,
        make_options(solver)
    );

    const std::vector<double> X{
        1.0,
        2.0,
        3.0
    };

    const std::vector<double> y{
        1.0,
        2.0
    };

    EXPECT_THROW(
        model.fit(X, y),
        std::invalid_argument
    );
}

TEST_P(LinearRegressionSolverTest, FitRejectsNanFeature)
{
    const auto solver = GetParam();

    LinearRegression model(
        1,
        make_options(solver)
    );

    const std::vector<double> X{
        1.0,
        std::numeric_limits<double>::quiet_NaN()
    };

    const std::vector<double> y{
        1.0,
        2.0
    };

    EXPECT_THROW(
        model.fit(X, y),
        std::invalid_argument
    );
}

TEST_P(LinearRegressionSolverTest, FitRejectsInfiniteFeature)
{
    const auto solver = GetParam();

    LinearRegression model(
        1,
        make_options(solver)
    );

    const std::vector<double> X{
        1.0,
        std::numeric_limits<double>::infinity()
    };

    const std::vector<double> y{
        1.0,
        2.0
    };

    EXPECT_THROW(
        model.fit(X, y),
        std::invalid_argument
    );
}

TEST_P(LinearRegressionSolverTest, FitRejectsNanResponse)
{
    const auto solver = GetParam();

    LinearRegression model(
        1,
        make_options(solver)
    );

    const std::vector<double> X{
        1.0,
        2.0
    };

    const std::vector<double> y{
        1.0,
        std::numeric_limits<double>::quiet_NaN()
    };

    EXPECT_THROW(
        model.fit(X, y),
        std::invalid_argument
    );
}

TEST_P(LinearRegressionSolverTest, FitRejectsInfiniteResponse)
{
    const auto solver = GetParam();

    LinearRegression model(
        1,
        make_options(solver)
    );

    const std::vector<double> X{
        1.0,
        2.0
    };

    const std::vector<double> y{
        1.0,
        std::numeric_limits<double>::infinity()
    };

    EXPECT_THROW(
        model.fit(X, y),
        std::invalid_argument
    );
}

TEST_P(LinearRegressionSolverTest, RankOneUpdateRejectsWrongDimension)
{
    const auto solver = GetParam();

    LinearRegression model(
        2,
        make_options(solver)
    );

    const std::array<double, 1> x{1.0};

    EXPECT_THROW(
        model.rk1_update(x, 1.0),
        std::invalid_argument
    );
}

TEST_P(LinearRegressionSolverTest, RankOneUpdateRejectsNanFeature)
{
    const auto solver = GetParam();

    LinearRegression model(
        2,
        make_options(solver)
    );

    const std::array<double, 2> x{
        1.0,
        std::numeric_limits<double>::quiet_NaN()
    };

    EXPECT_THROW(
        model.rk1_update(x, 1.0),
        std::invalid_argument
    );
}

TEST_P(LinearRegressionSolverTest, RankOneUpdateRejectsInfiniteResponse)
{
    const auto solver = GetParam();

    LinearRegression model(
        2,
        make_options(solver)
    );

    const std::array<double, 2> x{
        1.0,
        2.0
    };

    EXPECT_THROW(
        model.rk1_update(
            x,
            std::numeric_limits<double>::infinity()
        ),
        std::invalid_argument
    );
}

TEST_P(LinearRegressionSolverTest, BatchUpdateRejectsEmptyResponse)
{
    const auto solver = GetParam();

    LinearRegression model(
        2,
        make_options(solver)
    );

    const std::vector<double> X;
    const std::vector<double> y;

    EXPECT_THROW(
        model.batch_update(X, y),
        std::invalid_argument
    );
}

TEST_P(
    LinearRegressionSolverTest,
    BatchUpdateRejectsIncompatibleDimensions
)
{
    const auto solver = GetParam();

    LinearRegression model(
        2,
        make_options(solver)
    );

    const std::vector<double> X{
        1.0,
        2.0,
        3.0
    };

    const std::vector<double> y{
        1.0,
        2.0
    };

    EXPECT_THROW(
        model.batch_update(X, y),
        std::invalid_argument
    );
}

TEST_P(LinearRegressionSolverTest, PredictRejectsWrongDimension)
{
    const auto solver = GetParam();

    LinearRegression model(
        2,
        make_options(solver)
    );

    const std::vector<double> X = to_column_major({
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0}
    });

    const std::vector<double> y{
        1.0,
        2.0,
        3.0
    };

    model.fit(X, y);

    const std::array<double, 1> input{1.0};

    EXPECT_THROW(
        static_cast<void>(model.predict(input)),
        std::invalid_argument
    );
}

TEST_P(LinearRegressionSolverTest, PredictRejectsNonFiniteInput)
{
    const auto solver = GetParam();

    LinearRegression model(
        1,
        make_options(solver)
    );

    const std::vector<double> X{
        0.0,
        1.0
    };

    const std::vector<double> y{
        1.0,
        2.0
    };

    model.fit(X, y);

    const std::array<double, 1> input{
        std::numeric_limits<double>::quiet_NaN()
    };

    EXPECT_THROW(
        static_cast<void>(model.predict(input)),
        std::invalid_argument
    );
}

TEST_P(LinearRegressionSolverTest, AccessorsRejectUnfittedModel)
{
    const auto solver = GetParam();

    LinearRegression model(
        2,
        make_options(solver)
    );

    EXPECT_THROW(
        static_cast<void>(model.coefficients()),
        std::logic_error
    );

    EXPECT_THROW(
        static_cast<void>(model.intercept()),
        std::logic_error
    );

    const std::array<double, 2> input{
        1.0,
        2.0
    };

    EXPECT_THROW(
        static_cast<void>(model.predict(input)),
        std::logic_error
    );
}
/*
 * R-squared tests.
 */

TEST_P(
    LinearRegressionSolverTest,
    ExactFitHasUnitRSquared
)
{
    LinearRegression model(
        1,
        make_options(GetParam())
    );

    const std::vector<double> X{
        0.0,
        1.0,
        2.0,
        3.0
    };

    const std::vector<double> y{
        1.0,
        3.0,
        5.0,
        7.0
    };

    model.fit(X, y);

    EXPECT_NEAR(
        model.r_squared(),
        1.0,
        tolerance
    );
}

TEST_P(
    LinearRegressionSolverTest,
    NoisyFitHasKnownRSquared
)
{
    LinearRegression model(
        1,
        make_options(GetParam())
    );

    /*
     * The fitted line is
     *
     *     y_hat = 0.9 + 0.9x.
     *
     * RSS = 0.7 and TSS = 4.75, giving
     *
     *     R^2 = 1 - 0.7 / 4.75
     *         = 81 / 95.
     */
    const std::vector<double> X{
        0.0,
        1.0,
        2.0,
        3.0
    };

    const std::vector<double> y{
        1.0,
        2.0,
        2.0,
        4.0
    };

    model.fit(X, y);

    EXPECT_NEAR(
        model.r_squared(),
        81.0 / 95.0,
        tolerance
    );
}

TEST_P(
    LinearRegressionSolverTest,
    RidgeRSquaredExcludesPenalty
)
{
    LinearRegression model(
        1,
        make_options(
            GetParam(),
            true,
            3.0
        )
    );

    /*
     * The ridge solution is
     *
     *     intercept = 5,
     *     coefficient = 0.8.
     *
     * Predictions are (4.2, 5.0, 5.8), so
     *
     *     RSS = 2.88,
     *     TSS = 8,
     *     R^2 = 0.64.
     */
    const std::vector<double> X{
        -1.0,
         0.0,
         1.0
    };

    const std::vector<double> y{
        3.0,
        5.0,
        7.0
    };

    model.fit(X, y);

    EXPECT_NEAR(
        model.r_squared(),
        0.64,
        tolerance
    );
}

TEST_P(
    LinearRegressionSolverTest,
    NoInterceptModelUsesCentredTotalSumOfSquares
)
{
    LinearRegression model(
        1,
        make_options(
            GetParam(),
            false
        )
    );

    /*
     * Without an intercept:
     *
     *     beta = (1*2 + 2*1) / (1 + 4) = 0.8.
     *
     * RSS = 1.8 and centred TSS = 0.5, hence
     *
     *     R^2 = 1 - 1.8 / 0.5 = -2.6.
     */
    const std::vector<double> X{
        1.0,
        2.0
    };

    const std::vector<double> y{
        2.0,
        1.0
    };

    model.fit(X, y);

    EXPECT_NEAR(
        model.r_squared(),
        -2.6,
        tolerance
    );
}

TEST_P(
    LinearRegressionSolverTest,
    PerfectConstantResponseHasUnitRSquared
)
{
    LinearRegression model(
        1,
        make_options(GetParam())
    );

    const std::vector<double> X{
        0.0,
        1.0,
        2.0,
        3.0
    };

    const std::vector<double> y{
        5.0,
        5.0,
        5.0,
        5.0
    };

    model.fit(X, y);

    EXPECT_NEAR(
        model.r_squared(),
        1.0,
        tolerance
    );
}

TEST_P(
    LinearRegressionSolverTest,
    ImperfectConstantResponseHasZeroRSquared
)
{
    LinearRegression model(
        1,
        make_options(
            GetParam(),
            false
        )
    );

    const std::vector<double> X{
        1.0,
        2.0,
        3.0
    };

    const std::vector<double> y{
        5.0,
        5.0,
        5.0
    };

    model.fit(X, y);

    EXPECT_NEAR(
        model.r_squared(),
        0.0,
        tolerance
    );
}

TEST_P(
    LinearRegressionSolverTest,
    RankOneUpdateRefreshesRSquared
)
{
    LinearRegression updated_model(
        1,
        make_options(GetParam())
    );

    LinearRegression reference_model(
        1,
        make_options(GetParam())
    );

    const std::vector<double> initial_X{
        0.0,
        1.0,
        2.0
    };

    const std::vector<double> initial_y{
        1.0,
        3.0,
        5.0
    };

    updated_model.fit(initial_X, initial_y);

    EXPECT_NEAR(
        updated_model.r_squared(),
        1.0,
        tolerance
    );

    const std::array<double, 1> new_x{3.0};
    updated_model.rk1_update(new_x, 8.0);

    const std::vector<double> complete_X{
        0.0,
        1.0,
        2.0,
        3.0
    };

    const std::vector<double> complete_y{
        1.0,
        3.0,
        5.0,
        8.0
    };

    reference_model.fit(
        complete_X,
        complete_y
    );

    EXPECT_NEAR(
        updated_model.r_squared(),
        reference_model.r_squared(),
        tolerance
    );
}

TEST_P(
    LinearRegressionSolverTest,
    BatchUpdateRefreshesRSquared
)
{
    LinearRegression updated_model(
        1,
        make_options(GetParam())
    );

    LinearRegression reference_model(
        1,
        make_options(GetParam())
    );

    const std::vector<double> initial_X{
        0.0,
        1.0
    };

    const std::vector<double> initial_y{
        1.0,
        3.0
    };

    updated_model.fit(initial_X, initial_y);

    const std::vector<double> additional_X{
        2.0,
        3.0
    };

    const std::vector<double> additional_y{
        5.0,
        8.0
    };

    updated_model.batch_update(
        additional_X,
        additional_y
    );

    const std::vector<double> complete_X{
        0.0,
        1.0,
        2.0,
        3.0
    };

    const std::vector<double> complete_y{
        1.0,
        3.0,
        5.0,
        8.0
    };

    reference_model.fit(
        complete_X,
        complete_y
    );

    EXPECT_NEAR(
        updated_model.r_squared(),
        reference_model.r_squared(),
        tolerance
    );
}

TEST_P(
    LinearRegressionSolverTest,
    RefitReplacesResponseStatistics
)
{
    LinearRegression model(
        1,
        make_options(GetParam())
    );

    const std::vector<double> first_X{
        0.0,
        1.0,
        2.0
    };

    const std::vector<double> first_y{
        1.0,
        3.0,
        5.0
    };

    model.fit(first_X, first_y);

    EXPECT_NEAR(
        model.r_squared(),
        1.0,
        tolerance
    );

    const std::vector<double> second_X{
        0.0,
        1.0,
        2.0,
        3.0
    };

    const std::vector<double> second_y{
        1.0,
        2.0,
        2.0,
        4.0
    };

    model.fit(second_X, second_y);

    EXPECT_NEAR(
        model.r_squared(),
        81.0 / 95.0,
        tolerance
    );
}

TEST_P(
    LinearRegressionSolverTest,
    RSquaredRejectsUnfittedAndResetModel
)
{
    LinearRegression model(
        1,
        make_options(GetParam())
    );

    EXPECT_THROW(
        {
            const double result = model.r_squared();
            static_cast<void>(result);
        },
        std::logic_error
    );

    const std::vector<double> X{
        0.0,
        1.0,
        2.0
    };

    const std::vector<double> y{
        1.0,
        3.0,
        5.0
    };

    model.fit(X, y);
    model.reset();

    EXPECT_THROW(
        {
            const double result = model.r_squared();
            static_cast<void>(result);
        },
        std::logic_error
    );
}

TEST_P(
    LinearRegressionSolverTest,
    RSquaredIsStableForLargeResponseOffset
)
{
    LinearRegression model(
        1,
        make_options(GetParam())
    );

    /*
     * Adding a constant to every response must not change R^2.
     */
    constexpr double offset = 1e9;

    const std::vector<double> X{
        0.0,
        1.0,
        2.0,
        3.0
    };

    const std::vector<double> y{
        offset + 1.0,
        offset + 2.0,
        offset + 2.0,
        offset + 4.0
    };

    model.fit(X, y);

    EXPECT_NEAR(
        model.r_squared(),
        81.0 / 95.0,
        1e-6
    );
}

TEST(
    LinearRegressionRSquaredSvdTest,
    RankDeficientExactFitHasUnitRSquared
)
{
    LinearRegression model(
        2,
        make_options(
            LinearRegressionSolver::svd
        )
    );

    const std::vector<double> X = to_column_major({
        {-1.0, -2.0},
        { 0.0,  0.0},
        { 1.0,  2.0}
    });

    const std::vector<double> y{
        -3.0,
         0.0,
         3.0
    };

    model.fit(X, y);

    EXPECT_NEAR(
        model.r_squared(),
        1.0,
        1e-8
    );
}
}