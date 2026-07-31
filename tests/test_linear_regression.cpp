
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