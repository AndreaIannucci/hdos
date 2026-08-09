#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "hdos/online_moments.hpp"

namespace {

constexpr double tol = 1e-12;

TEST(RunningMeanTest, SequentialUpdate)
{
    hdos::RunningMean mean(2);

    std::vector<double> x1{1.0, 10.0};
    std::vector<double> x2{2.0, 20.0};
    std::vector<double> x3{3.0, 30.0};
    std::vector<double> x4{4.0, 40.0};

    mean.update(x1);
    mean.update(x2);
    mean.update(x3);
    mean.update(x4);

    EXPECT_EQ(mean.n_observations(), 4);
    EXPECT_EQ(mean.n_features(), 2);

    EXPECT_NEAR(mean.mean()[0], 2.5, tol);
    EXPECT_NEAR(mean.mean()[1], 25.0, tol);
}


TEST(RunningMeanTest, BatchUpdate)
{
    hdos::RunningMean mean(2);

    // (1,10), (2,20), (3,30), (4,40)
    std::vector<double> X{
         1.0,  2.0,  3.0,  4.0,
        10.0, 20.0, 30.0, 40.0
    };

    mean.batch_update(X);

    EXPECT_EQ(mean.n_observations(), 4);

    EXPECT_NEAR(mean.mean()[0], 2.5, tol);
    EXPECT_NEAR(mean.mean()[1], 25.0, tol);
}


TEST(RunningMeanTest, BatchMatchesSequentialUpdates)
{
    hdos::RunningMean batch(2);
    hdos::RunningMean sequential(2);

    std::vector<double> X{
         1.0,  2.0,  3.0,  4.0,
        10.0, 20.0, 30.0, 40.0
    };

    batch.batch_update(X);

    for (std::size_t i = 0; i < 4; ++i) {
        std::vector<double> x{
            X[i],
            X[i + 4]
        };

        sequential.update(x);
    }

    EXPECT_EQ(
        batch.n_observations(),
        sequential.n_observations());

    for (std::size_t j = 0; j < 2; ++j) {
        EXPECT_NEAR(
            batch.mean()[j],
            sequential.mean()[j],
            tol);
    }
}


TEST(RunningMeanTest, BatchUpdateAfterExistingObservations)
{
    hdos::RunningMean mean(2);

    std::vector<double> x{0.0, 10.0};
    mean.update(x);

    // (2,20), (4,30), (6,40)
    std::vector<double> X{
         2.0,  4.0,  6.0,
        20.0, 30.0, 40.0
    };

    mean.batch_update(X);

    EXPECT_EQ(mean.n_observations(), 4);

    EXPECT_NEAR(mean.mean()[0], 3.0, tol);
    EXPECT_NEAR(mean.mean()[1], 25.0, tol);
}


TEST(RunningMeanTest, Reset)
{
    hdos::RunningMean mean(2);

    std::vector<double> x{3.0, 7.0};
    mean.update(x);

    mean.reset();

    EXPECT_EQ(mean.n_observations(), 0);
    EXPECT_NEAR(mean.mean()[0], 0.0, tol);
    EXPECT_NEAR(mean.mean()[1], 0.0, tol);
}


TEST(RunningMeanTest, EmptyBatchDoesNothing)
{
    hdos::RunningMean mean(2);

    std::vector<double> x{2.0, 4.0};
    mean.update(x);

    std::vector<double> empty;
    mean.batch_update(empty);

    EXPECT_EQ(mean.n_observations(), 1);
    EXPECT_NEAR(mean.mean()[0], 2.0, tol);
    EXPECT_NEAR(mean.mean()[1], 4.0, tol);
}


TEST(RunningMeanTest, RejectsInvalidDimension)
{
    hdos::RunningMean mean(2);

    std::vector<double> x{1.0, 2.0, 3.0};

    EXPECT_THROW(
        mean.update(x),
        std::invalid_argument);

    EXPECT_EQ(mean.n_observations(), 0);
}


TEST(RunningMeanTest, RejectsInvalidBatchDimensions)
{
    hdos::RunningMean mean(2);

    std::vector<double> X{
        1.0, 2.0, 3.0
    };

    EXPECT_THROW(
        mean.batch_update(X),
        std::invalid_argument);

    EXPECT_EQ(mean.n_observations(), 0);
}


TEST(RunningMeanTest, RejectsNaN)
{
    hdos::RunningMean mean(2);

    std::vector<double> x{
        std::numeric_limits<double>::quiet_NaN(),
        1.0
    };

    EXPECT_THROW(
        mean.update(x),
        std::invalid_argument);

    EXPECT_EQ(mean.n_observations(), 0);
}

}