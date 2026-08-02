
#include <gtest/gtest.h>

#include "hdos/linear_regression.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

TEST(LinearRegression, RecoversExactModelWithIntercept)
{
    hdos::LinearRegressionOptions options;
    options.fit_intercept = true;

    hdos::LinearRegression model(2, options);

    // Rows:
    //
    // [0  1]
    // [1  0]
    // [2  1]
    // [3  2]
    //
    // Stored column-major.
    const std::vector<double> X{
        0.0, 1.0, 2.0, 3.0,
        1.0, 0.0, 1.0, 2.0
    };

    // y = 3 + 2*x_1 - x_2.
    const std::vector<double> y{
        2.0, 5.0, 6.0, 7.0
    };

    model.fit(X, y);

    const std::vector<double>& coefficients =
        model.coefficients();

    ASSERT_EQ(coefficients.size(), 2U);
    EXPECT_NEAR(coefficients[0],  2.0, 1e-10);
    EXPECT_NEAR(coefficients[1], -1.0, 1e-10);
    EXPECT_NEAR(model.intercept(), 3.0, 1e-10);
}

TEST(LinearRegression, RecoversExactModelWithoutIntercept)
{
    hdos::LinearRegressionOptions options;
    options.fit_intercept = false;

    hdos::LinearRegression model(2, options);

    const std::vector<double> X{
        0.0, 1.0, 2.0, 3.0,
        1.0, 0.0, 1.0, 2.0
    };

    // y = 2*x_1 - x_2.
    const std::vector<double> y{
        -1.0, 2.0, 3.0, 4.0
    };

    model.fit(X, y);

    const std::vector<double>& coefficients =
        model.coefficients();

    ASSERT_EQ(coefficients.size(), 2U);
    EXPECT_NEAR(coefficients[0],  2.0, 1e-10);
    EXPECT_NEAR(coefficients[1], -1.0, 1e-10);
    EXPECT_NEAR(model.intercept(), 0.0, 1e-10);
}



TEST(LinearRegression, RecoversLeastSquaresFitWithNoise)
{
    hdos::LinearRegression model(1);

    const std::vector<double> X{
        -2.0, -1.0, 0.0, 1.0, 2.0
    };

    // OLS slope = 2 and intercept = 1, despite the residual noise.
    const std::vector<double> y{
        -2.0, -2.0, 1.0, 2.0, 6.0
    };

    model.fit(X, y);

    // Calling intercept first also tests that either accessor
    // can trigger the lazy solution.
    EXPECT_NEAR(model.intercept(), 1.0, 1e-10);

    const std::vector<double>& coefficients =
        model.coefficients();

    ASSERT_EQ(coefficients.size(), 1U);
    EXPECT_NEAR(coefficients[0], 2.0, 1e-10);
}

TEST(LinearRegression, RefitInvalidatesPreviousSolution)
{
    hdos::LinearRegressionOptions options;
    options.fit_intercept = false;

    hdos::LinearRegression model(1, options);

    const std::vector<double> X{
        1.0, 2.0, 3.0, 4.0
    };

    const std::vector<double> first_y{
        2.0, 4.0, 6.0, 8.0
    };

    model.fit(X, first_y);

    ASSERT_EQ(model.coefficients().size(), 1U);
    EXPECT_NEAR(model.coefficients()[0], 2.0, 1e-10);

    const std::vector<double> second_y{
        -1.0, -2.0, -3.0, -4.0
    };

    model.fit(X, second_y);

    ASSERT_EQ(model.coefficients().size(), 1U);
    EXPECT_NEAR(model.coefficients()[0], -1.0, 1e-10);
}


TEST(LinearRegression, RejectsCoefficientAccessBeforeFit)
{
    hdos::LinearRegression model(2);

    EXPECT_THROW(
        model.coefficients(),
        std::logic_error
    );

    EXPECT_THROW(
        model.intercept(),
        std::logic_error
    );
}

TEST(LinearRegression, RejectsRankDeficientDesign)
{
    hdos::LinearRegressionOptions options;
    options.fit_intercept = false;

    hdos::LinearRegression model(2, options);

    // Two identical columns:
    //
    // [1  1]
    // [0  0]
    const std::vector<double> X{
        1.0, 0.0,
        1.0, 0.0
    };

    const std::vector<double> y{
        1.0, 0.0
    };

    EXPECT_THROW(
        model.fit(X, y),
        std::invalid_argument
    );
}




TEST(LinearRegressionRankOneUpdate, MatchesRefitWithoutIntercept)
{
    hdos::LinearRegressionOptions options;
    options.fit_intercept = false;

    hdos::LinearRegression incremental_model(2, options);
    hdos::LinearRegression refitted_model(2, options);

    // Initial observations:
    //
    // X = [1  0]
    //     [0  1]
    //
    // y = [1, 2]
    const std::vector<double> initial_X{
        1.0, 0.0,
        0.0, 1.0
    };

    const std::vector<double> initial_y{
        1.0, 2.0
    };

    incremental_model.fit(initial_X, initial_y);

    // Force the initial lazy solve.
    const auto& initial_coefficients =
        incremental_model.coefficients();

    ASSERT_EQ(initial_coefficients.size(), 2U);
    EXPECT_NEAR(initial_coefficients[0], 1.0, 1e-12);
    EXPECT_NEAR(initial_coefficients[1], 2.0, 1e-12);

    // Add x = [1,1], y = 6.
    const std::vector<double> new_x{
        1.0, 1.0
    };

    incremental_model.rk1_update(new_x, 6.0);

    // Refit on all three observations.
    //
    // X = [1  0]
    //     [0  1]
    //     [1  1]
    //
    // Column-major storage:
    const std::vector<double> complete_X{
        1.0, 0.0, 1.0,
        0.0, 1.0, 1.0
    };

    const std::vector<double> complete_y{
        1.0, 2.0, 6.0
    };

    refitted_model.fit(complete_X, complete_y);

    const auto& incremental_coefficients =
        incremental_model.coefficients();

    const auto& refitted_coefficients =
        refitted_model.coefficients();

    ASSERT_EQ(incremental_coefficients.size(), 2U);
    ASSERT_EQ(refitted_coefficients.size(), 2U);

    // The exact updated solution is [2,3].
    EXPECT_NEAR(incremental_coefficients[0], 2.0, 1e-12);
    EXPECT_NEAR(incremental_coefficients[1], 3.0, 1e-12);

    EXPECT_NEAR(
        incremental_coefficients[0],
        refitted_coefficients[0],
        1e-12
    );

    EXPECT_NEAR(
        incremental_coefficients[1],
        refitted_coefficients[1],
        1e-12
    );

    EXPECT_NEAR(incremental_model.intercept(), 0.0, 1e-12);
    EXPECT_EQ(incremental_model.n_observations(), 3U);
}


TEST(LinearRegressionRankOneUpdate, MatchesRefitWithIntercept)
{
    hdos::LinearRegressionOptions options;
    options.fit_intercept = true;

    hdos::LinearRegression incremental_model(1, options);
    hdos::LinearRegression refitted_model(1, options);

    // Initial data:
    //
    // x = [0,1]
    // y = [1,3]
    const std::vector<double> initial_X{
        0.0, 1.0
    };

    const std::vector<double> initial_y{
        1.0, 3.0
    };

    incremental_model.fit(initial_X, initial_y);

    // Force the initial lazy solve.
    const auto& initial_coefficients =
        incremental_model.coefficients();

    ASSERT_EQ(initial_coefficients.size(), 1U);
    EXPECT_NEAR(initial_coefficients[0], 2.0, 1e-12);
    EXPECT_NEAR(incremental_model.intercept(), 1.0, 1e-12);

    // Add x = 2, y = 10.
    const std::vector<double> new_x{
        2.0
    };

    incremental_model.rk1_update(new_x, 10.0);

    const std::vector<double> complete_X{
        0.0, 1.0, 2.0
    };

    const std::vector<double> complete_y{
        1.0, 3.0, 10.0
    };

    refitted_model.fit(complete_X, complete_y);

    const auto& incremental_coefficients =
        incremental_model.coefficients();

    const auto& refitted_coefficients =
        refitted_model.coefficients();

    ASSERT_EQ(incremental_coefficients.size(), 1U);
    ASSERT_EQ(refitted_coefficients.size(), 1U);

    // The exact least-squares solution is:
    //
    // slope     = 9/2
    // intercept = 1/6
    EXPECT_NEAR(
        incremental_coefficients[0],
        4.5,
        1e-12
    );

    EXPECT_NEAR(
        incremental_model.intercept(),
        1.0 / 6.0,
        1e-12
    );

    EXPECT_NEAR(
        incremental_coefficients[0],
        refitted_coefficients[0],
        1e-12
    );

    EXPECT_NEAR(
        incremental_model.intercept(),
        refitted_model.intercept(),
        1e-12
    );

    EXPECT_EQ(incremental_model.n_observations(), 3U);
}













TEST(LinearRegressionBatchUpdate, MatchesSequentialRankOneUpdates)
{
    hdos::LinearRegressionOptions options;
    options.fit_intercept = false;

    hdos::LinearRegression sequential_model(2, options);
    hdos::LinearRegression batch_model(2, options);
    hdos::LinearRegression refitted_model(2, options);

    const std::vector<double> initial_X{
        1.0, 0.0,
        0.0, 1.0
    };

    const std::vector<double> initial_y{
        1.0, 2.0
    };

    sequential_model.fit(initial_X, initial_y);
    batch_model.fit(initial_X, initial_y);

    const std::vector<double> x1{
        1.0, 1.0
    };

    const std::vector<double> x2{
        2.0, -1.0
    };

    sequential_model.rk1_update(x1, 6.0);
    sequential_model.rk1_update(x2, 0.0);

    // Two observations stored column-major:
    //
    // [1   1]
    // [2  -1]
    const std::vector<double> batch_X{
        1.0, 2.0,
        1.0, -1.0
    };

    const std::vector<double> batch_y{
        6.0, 0.0
    };

    batch_model.batch_update(batch_X, batch_y);

    // All four observations stored column-major.
    const std::vector<double> complete_X{
        1.0, 0.0, 1.0, 2.0,
        0.0, 1.0, 1.0, -1.0
    };

    const std::vector<double> complete_y{
        1.0, 2.0, 6.0, 0.0
    };

    refitted_model.fit(complete_X, complete_y);

    const auto& sequential_coefficients =
        sequential_model.coefficients();

    const auto& batch_coefficients =
        batch_model.coefficients();

    const auto& refitted_coefficients =
        refitted_model.coefficients();

    ASSERT_EQ(sequential_coefficients.size(), 2U);
    ASSERT_EQ(batch_coefficients.size(), 2U);
    ASSERT_EQ(refitted_coefficients.size(), 2U);

    // Exact solution:
    //
    // beta_1 = 29/17
    // beta_2 = 55/17
    const std::vector<double> expected{
        29.0 / 17.0,
        55.0 / 17.0
    };

    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(
            sequential_coefficients[i],
            expected[i],
            1e-12
        );

        EXPECT_NEAR(
            batch_coefficients[i],
            expected[i],
            1e-12
        );

        EXPECT_NEAR(
            refitted_coefficients[i],
            expected[i],
            1e-12
        );

        EXPECT_NEAR(
            sequential_coefficients[i],
            batch_coefficients[i],
            1e-12
        );
    }

    EXPECT_EQ(sequential_model.n_observations(), 4U);
    EXPECT_EQ(batch_model.n_observations(), 4U);
    EXPECT_EQ(refitted_model.n_observations(), 4U);
}