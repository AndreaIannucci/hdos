#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "hdos/online_moments.hpp"

namespace {

constexpr double tol = 1e-12;

TEST(RunningVarianceTest, SequentialUpdate)
{
    hdos::RunningVariance moments(2);

    std::vector<double> x1{1.0, 2.0};
    std::vector<double> x2{2.0, 4.0};
    std::vector<double> x3{3.0, 6.0};
    std::vector<double> x4{4.0, 8.0};

    moments.update(x1);
    moments.update(x2);
    moments.update(x3);
    moments.update(x4);

    EXPECT_EQ(moments.n_observations(), 4);

    EXPECT_NEAR(moments.mean()[0], 2.5, tol);
    EXPECT_NEAR(moments.mean()[1], 5.0, tol);

    // Sample variances.
    EXPECT_NEAR(
        moments.variance()[0],
        5.0 / 3.0,
        tol);

    EXPECT_NEAR(
        moments.variance()[1],
        20.0 / 3.0,
        tol);
}


TEST(RunningVarianceTest, SingleObservationHasZeroVariance)
{
    hdos::RunningVariance moments(2);

    std::vector<double> x{3.0, 7.0};

    moments.update(x);

    EXPECT_EQ(moments.n_observations(), 1);

    EXPECT_NEAR(moments.variance()[0], 0.0, tol);
    EXPECT_NEAR(moments.variance()[1], 0.0, tol);
}


TEST(RunningVarianceTest, BatchUpdate)
{
    hdos::RunningVariance moments(2);

    std::vector<double> X{
        1.0, 2.0, 3.0, 4.0,
        2.0, 4.0, 6.0, 8.0
    };

    moments.batch_update(X);

    EXPECT_EQ(moments.n_observations(), 4);

    EXPECT_NEAR(moments.mean()[0], 2.5, tol);
    EXPECT_NEAR(moments.mean()[1], 5.0, tol);

    EXPECT_NEAR(
        moments.variance()[0],
        5.0 / 3.0,
        tol);

    EXPECT_NEAR(
        moments.variance()[1],
        20.0 / 3.0,
        tol);
}


TEST(RunningVarianceTest, BatchMatchesSequentialUpdates)
{
    hdos::RunningVariance batch(2);
    hdos::RunningVariance sequential(2);

    std::vector<double> X{
        1.0, 2.0, 3.0, 4.0,
        2.0, 4.0, 6.0, 8.0
    };

    batch.batch_update(X);

    for (std::size_t i = 0; i < 4; ++i) {

        std::vector<double> x{
            X[i],
            X[i + 4]
        };

        sequential.update(x);
    }

    for (std::size_t j = 0; j < 2; ++j) {

        EXPECT_NEAR(
            batch.mean()[j],
            sequential.mean()[j],
            tol);

        EXPECT_NEAR(
            batch.variance()[j],
            sequential.variance()[j],
            tol);
    }
}


TEST(RunningVarianceTest, BatchMergeWithExistingState)
{
    hdos::RunningVariance batch(2);
    hdos::RunningVariance sequential(2);

    std::vector<double> x0{0.0, 0.0};

    batch.update(x0);
    sequential.update(x0);

    std::vector<double> X{
        1.0, 2.0, 3.0,
        2.0, 4.0, 6.0
    };

    batch.batch_update(X);

    std::vector<double> x1{1.0, 2.0};
    std::vector<double> x2{2.0, 4.0};
    std::vector<double> x3{3.0, 6.0};

    sequential.update(x1);
    sequential.update(x2);
    sequential.update(x3);

    EXPECT_EQ(batch.n_observations(), 4);

    for (std::size_t j = 0; j < 2; ++j) {
        EXPECT_NEAR(
            batch.mean()[j],
            sequential.mean()[j],
            tol);

        EXPECT_NEAR(
            batch.variance()[j],
            sequential.variance()[j],
            tol);
    }
}


TEST(RunningVarianceTest, ConstantDataHasZeroVariance)
{
    hdos::RunningVariance moments(2);

    std::vector<double> X{
        3.0, 3.0, 3.0, 3.0,
       -2.0,-2.0,-2.0,-2.0
    };

    moments.batch_update(X);

    EXPECT_NEAR(moments.variance()[0], 0.0, tol);
    EXPECT_NEAR(moments.variance()[1], 0.0, tol);
}


TEST(RunningVarianceTest, Reset)
{
    hdos::RunningVariance moments(2);

    std::vector<double> X{
        1.0, 2.0, 3.0,
        2.0, 4.0, 6.0
    };

    moments.batch_update(X);
    moments.reset();

    EXPECT_EQ(moments.n_observations(), 0);

    for (double value : moments.mean()) {
        EXPECT_NEAR(value, 0.0, tol);
    }

    for (double value : moments.variance()) {
        EXPECT_NEAR(value, 0.0, tol);
    }
}


TEST(RunningVarianceTest, StableForLargeOffset)
{
    hdos::RunningVariance moments(1);

    std::vector<double> X{
        1e12 + 1.0,
        1e12 + 2.0,
        1e12 + 3.0,
        1e12 + 4.0
    };

    moments.batch_update(X);

    EXPECT_NEAR(
        moments.mean()[0],
        1e12 + 2.5,
        1e-6);

    EXPECT_NEAR(
        moments.variance()[0],
        5.0 / 3.0,
        1e-9);
}


TEST(RunningVarianceTest, RejectsNaN)
{
    hdos::RunningVariance moments(2);

    std::vector<double> x{
        std::numeric_limits<double>::quiet_NaN(),
        1.0
    };

    EXPECT_THROW(
        moments.update(x),
        std::invalid_argument);

    EXPECT_EQ(moments.n_observations(), 0);
}

}