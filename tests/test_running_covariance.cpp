#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "hdos/online_moments.hpp"

namespace {

constexpr double tol = 1e-12;

void expect_near(
    std::span<const double> actual,
    const std::vector<double>& expected,
    double tolerance = tol)
{
    ASSERT_EQ(actual.size(), expected.size());

    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(
            actual[i],
            expected[i],
            tolerance);
    }
}


TEST(RunningCovarianceTest, SequentialUpdate)
{
    hdos::RunningCovariance moments(2);

    moments.update(std::vector<double>{1.0, 2.0});
    moments.update(std::vector<double>{2.0, 4.0});
    moments.update(std::vector<double>{3.0, 6.0});
    moments.update(std::vector<double>{4.0, 8.0});

    EXPECT_EQ(moments.n_observations(), 4);

    EXPECT_NEAR(moments.mean()[0], 2.5, tol);
    EXPECT_NEAR(moments.mean()[1], 5.0, tol);

    // Column-major representation of:
    //
    // [  5/3   10/3 ]
    // [ 10/3   20/3 ]
    const std::vector<double> expected{
        5.0 / 3.0,
        10.0 / 3.0,
        10.0 / 3.0,
        20.0 / 3.0
    };

    expect_near(moments.covariance(), expected);
}


TEST(RunningCovarianceTest, SingleObservationHasZeroCovariance)
{
    hdos::RunningCovariance moments(2);

    moments.update(std::vector<double>{3.0, 7.0});

    EXPECT_EQ(moments.n_observations(), 1);

    const std::vector<double> expected{
        0.0, 0.0,
        0.0, 0.0
    };

    expect_near(moments.covariance(), expected);
}


TEST(RunningCovarianceTest, BatchUpdate)
{
    hdos::RunningCovariance moments(3);

    // Three observations, stored column-major:
    //
    // [1, 2, 3]
    // [2, 0, 4]
    // [4, 1, 2]
    std::vector<double> X{
        1.0, 2.0, 4.0,
        2.0, 0.0, 1.0,
        3.0, 4.0, 2.0
    };

    moments.batch_update(X);

    EXPECT_EQ(moments.n_observations(), 3);

    EXPECT_NEAR(moments.mean()[0], 7.0 / 3.0, tol);
    EXPECT_NEAR(moments.mean()[1], 1.0, tol);
    EXPECT_NEAR(moments.mean()[2], 3.0, tol);

    // Sample covariance:
    //
    // [ 7/3  -1/2  -1   ]
    // [-1/2   1    -1/2 ]
    // [-1    -1/2   1   ]
    const std::vector<double> expected{
        7.0 / 3.0, -0.5, -1.0,
        -0.5,         1.0, -0.5,
        -1.0,        -0.5,  1.0
    };

    expect_near(moments.covariance(), expected);
}


TEST(RunningCovarianceTest, BatchMatchesSequentialUpdates)
{
    hdos::RunningCovariance batch(2);
    hdos::RunningCovariance sequential(2);

    std::vector<double> X{
        1.0, 2.0, 3.0, 4.0,
        2.0, 4.0, 1.0, 8.0
    };

    batch.batch_update(X);

    for (std::size_t i = 0; i < 4; ++i) {
        sequential.update(
            std::vector<double>{
                X[i],
                X[i + 4]
            });
    }

    EXPECT_EQ(
        batch.n_observations(),
        sequential.n_observations());

    expect_near(
        batch.mean(),
        std::vector<double>(
            sequential.mean().begin(),
            sequential.mean().end()));

    expect_near(
        batch.covariance(),
        std::vector<double>(
            sequential.covariance().begin(),
            sequential.covariance().end()));
}


TEST(RunningCovarianceTest, BatchMergeWithExistingState)
{
    hdos::RunningCovariance batch(2);
    hdos::RunningCovariance sequential(2);

    const std::vector<double> initial{10.0, -1.0};

    batch.update(initial);
    sequential.update(initial);

    // Three additional observations:
    //
    // [1, 2]
    // [2, 4]
    // [3, 1]
    std::vector<double> X{
        1.0, 2.0, 3.0,
        2.0, 4.0, 1.0
    };

    batch.batch_update(X);

    sequential.update(std::vector<double>{1.0, 2.0});
    sequential.update(std::vector<double>{2.0, 4.0});
    sequential.update(std::vector<double>{3.0, 1.0});

    EXPECT_EQ(batch.n_observations(), 4);

    expect_near(
        batch.mean(),
        std::vector<double>(
            sequential.mean().begin(),
            sequential.mean().end()));

    expect_near(
        batch.covariance(),
        std::vector<double>(
            sequential.covariance().begin(),
            sequential.covariance().end()));
}


TEST(RunningCovarianceTest, ConstantDataHasZeroCovariance)
{
    hdos::RunningCovariance moments(2);

    std::vector<double> X{
         3.0,  3.0,  3.0,  3.0,
        -2.0, -2.0, -2.0, -2.0
    };

    moments.batch_update(X);

    const std::vector<double> expected{
        0.0, 0.0,
        0.0, 0.0
    };

    expect_near(moments.covariance(), expected);
}


TEST(RunningCovarianceTest, CacheIsInvalidatedByUpdate)
{
    hdos::RunningCovariance moments(2);

    moments.update(std::vector<double>{1.0, 2.0});
    moments.update(std::vector<double>{2.0, 4.0});

    const std::vector<double> before{
        moments.covariance().begin(),
        moments.covariance().end()
    };

    expect_near(
        before,
        std::vector<double>{
            0.5, 1.0,
            1.0, 2.0
        });

    moments.update(std::vector<double>{3.0, 6.0});

    const std::vector<double> expected_after{
        1.0, 2.0,
        2.0, 4.0
    };

    expect_near(
        moments.covariance(),
        expected_after);
}


TEST(RunningCovarianceTest, Reset)
{
    hdos::RunningCovariance moments(2);

    std::vector<double> X{
        1.0, 2.0, 3.0,
        2.0, 4.0, 6.0
    };

    moments.batch_update(X);

    // Populate the lazy cache before resetting.
    static_cast<void>(moments.covariance());

    moments.reset();

    EXPECT_EQ(moments.n_observations(), 0);

    expect_near(
        moments.mean(),
        std::vector<double>{0.0, 0.0});

    expect_near(
        moments.covariance(),
        std::vector<double>{
            0.0, 0.0,
            0.0, 0.0
        });
}


TEST(RunningCovarianceTest, StableForLargeOffset)
{
    hdos::RunningCovariance moments(2);

    std::vector<double> X{
         1e12 + 1.0,
         1e12 + 2.0,
         1e12 + 3.0,
         1e12 + 4.0,

        -1e12 + 2.0,
        -1e12 + 4.0,
        -1e12 + 6.0,
        -1e12 + 8.0
    };

    moments.batch_update(X);

    EXPECT_NEAR(
        moments.mean()[0],
        1e12 + 2.5,
        1e-6);

    EXPECT_NEAR(
        moments.mean()[1],
        -1e12 + 5.0,
        1e-6);

    const std::vector<double> expected{
        5.0 / 3.0,
        10.0 / 3.0,
        10.0 / 3.0,
        20.0 / 3.0
    };

    expect_near(
        moments.covariance(),
        expected,
        1e-9);
}


TEST(RunningCovarianceTest, RejectsNaN)
{
    hdos::RunningCovariance moments(2);

    std::vector<double> x{
        std::numeric_limits<double>::quiet_NaN(),
        1.0
    };

    EXPECT_THROW(
        moments.update(x),
        std::invalid_argument);

    EXPECT_EQ(moments.n_observations(), 0);
}


TEST(RunningCovarianceTest, RejectsInfinity)
{
    hdos::RunningCovariance moments(2);

    std::vector<double> X{
        1.0, 2.0,
        std::numeric_limits<double>::infinity(),
        4.0
    };

    EXPECT_THROW(
        moments.batch_update(X),
        std::invalid_argument);

    EXPECT_EQ(moments.n_observations(), 0);
}


TEST(RunningCovarianceTest, RejectsInvalidDimensions)
{
    hdos::RunningCovariance moments(2);

    EXPECT_THROW(
        moments.update(std::vector<double>{1.0}),
        std::invalid_argument);

    EXPECT_THROW(
        moments.batch_update(
            std::vector<double>{1.0, 2.0, 3.0}),
        std::invalid_argument);

    EXPECT_EQ(moments.n_observations(), 0);
}


TEST(RunningCovarianceTest, RejectsZeroFeatures)
{
    EXPECT_THROW(
        hdos::RunningCovariance(0),
        std::invalid_argument);
}

} 