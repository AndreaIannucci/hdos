#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "hdos/pca.hpp"

namespace {

constexpr double tol = 1e-10;

double abs_dot(
    std::span<const double> x,
    std::span<const double> y)
{
    double out = 0.0;

    for (std::size_t i = 0; i < x.size(); ++i) {
        out += x[i] * y[i];
    }

    return std::abs(out);
}

} 


TEST(PCATest, FitRankOneData)
{
    hdos::PCA pca(2, 1);

    // Observations:
    // (1,1), (2,2), (3,3), (4,4)
    //
    // Column-major storage.
    std::vector<double> X{
        1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0
    };

    pca.fit(X);

    const auto mean = pca.mean();
    const auto components = pca.components();
    const auto singular_values = pca.singular_values();
    const auto explained_variance = pca.explained_variance();

    ASSERT_EQ(mean.size(), 2);
    ASSERT_EQ(components.size(), 2);
    ASSERT_EQ(singular_values.size(), 1);
    ASSERT_EQ(explained_variance.size(), 1);

    EXPECT_NEAR(mean[0], 2.5, tol);
    EXPECT_NEAR(mean[1], 2.5, tol);

    // Sign of a singular vector is arbitrary.
    const double inv_sqrt_2 = 1.0 / std::sqrt(2.0);

    EXPECT_NEAR(std::abs(components[0]), inv_sqrt_2, tol);
    EXPECT_NEAR(std::abs(components[1]), inv_sqrt_2, tol);

    EXPECT_NEAR(singular_values[0], std::sqrt(10.0), tol);

    // lambda = sigma^2 / (n - 1) = 10 / 3.
    EXPECT_NEAR(explained_variance[0], 10.0 / 3.0, tol);

    EXPECT_EQ(pca.n_observations(), 4);
}



TEST(PCATest, UpdateInsideCurrentSubspace)
{
    hdos::PCA pca(2, 1);

    std::vector<double> X{
        1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0
    };

    pca.fit(X);

    std::vector<double> x{5.0, 5.0};
    pca.update(x);

    const auto mean = pca.mean();
    const auto components = pca.components();
    const auto singular_values = pca.singular_values();
    const auto explained_variance = pca.explained_variance();

    EXPECT_EQ(pca.n_observations(), 5);

    EXPECT_NEAR(mean[0], 3.0, tol);
    EXPECT_NEAR(mean[1], 3.0, tol);

    const double inv_sqrt_2 = 1.0 / std::sqrt(2.0);

    EXPECT_NEAR(std::abs(components[0]), inv_sqrt_2, tol);
    EXPECT_NEAR(std::abs(components[1]), inv_sqrt_2, tol);

    EXPECT_NEAR(singular_values[0], std::sqrt(20.0), tol);
    EXPECT_NEAR(explained_variance[0], 5.0, tol);
}

TEST(PCATest, UpdateMatchesBatchFit)
{
    hdos::PCA incremental(2, 1);

    std::vector<double> X{
        1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0
    };

    incremental.fit(X);

    std::vector<double> x{5.0, 2.0};
    incremental.update(x);


    // Same five observations fitted from scratch:
    //
    // (1,1), (2,2), (3,3), (4,4), (5,2)
    std::vector<double> X_full{
        1.0, 2.0, 3.0, 4.0, 5.0,
        1.0, 2.0, 3.0, 4.0, 2.0
    };

    hdos::PCA batch(2, 1);
    batch.fit(X_full);

    const auto mean_inc = incremental.mean();
    const auto mean_batch = batch.mean();

    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(mean_inc[i], mean_batch[i], tol);
    }

    EXPECT_NEAR(
        incremental.singular_values()[0],
        batch.singular_values()[0],
        tol
    );

    EXPECT_NEAR(
        incremental.explained_variance()[0],
        batch.explained_variance()[0],


        
        tol
    );

    // Compare directions up to their arbitrary sign.
    EXPECT_NEAR(
        abs_dot(
            incremental.components(),
            batch.components()
        ),
        1.0,
        tol
    );

    EXPECT_EQ(incremental.n_observations(), 5);
}


TEST(PCATest, UpdateBeforeFitThrows)
{
    hdos::PCA pca(2, 1);

    std::vector<double> x{1.0, 2.0};

    EXPECT_THROW(
        pca.update(x),
        std::logic_error
    );
}



TEST(PCATest, UpdateOutsideCurrentSubspace)
{
    hdos::PCA pca(2, 1);

    // Observations:
    // (1,0), (2,0), (3,0), (4,0)
    std::vector<double> X{
        1.0, 2.0, 3.0, 4.0,
        0.0, 0.0, 0.0, 0.0
    };

    pca.fit(X);

    // New observation is entirely outside the current PC direction.
    std::vector<double> x{2.5, 4.0};
    pca.update(x);

    EXPECT_EQ(pca.n_observations(), 5);

    const auto mean = pca.mean();
    EXPECT_NEAR(mean[0], 2.5, tol);
    EXPECT_NEAR(mean[1], 0.8, tol);

    const auto components = pca.components();

    // New dominant direction should be the y-axis.
    EXPECT_NEAR(std::abs(components[0]), 0.0, tol);
    EXPECT_NEAR(std::abs(components[1]), 1.0, tol);

    const auto singular_values = pca.singular_values();
    EXPECT_NEAR(
        singular_values[0],
        std::sqrt(12.8),
        tol);

    const auto explained_variance =
        pca.explained_variance();

    EXPECT_NEAR(explained_variance[0], 3.2, tol);
}





TEST(PCATest, ObservationCountAfterMultipleUpdates)
{
    hdos::PCA pca(2, 1);

    std::vector<double> X{
        1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0
    };

    pca.fit(X);

    EXPECT_EQ(pca.n_observations(), 4);

    std::vector<double> x1{5.0, 5.0};
    std::vector<double> x2{6.0, 6.0};

    pca.update(x1);
    EXPECT_EQ(pca.n_observations(), 5);

    pca.update(x2);
    EXPECT_EQ(pca.n_observations(), 6);

    const auto mean = pca.mean();

    EXPECT_NEAR(mean[0], 3.5, tol);
    EXPECT_NEAR(mean[1], 3.5, tol);
}



TEST(PCATest, BatchUpdateMatchesRepeatedUpdate)
{
    hdos::PCA batch_pca(2, 1);
    hdos::PCA sequential_pca(2, 1);

    std::vector<double> X{
        1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0
    };

    batch_pca.fit(X);
    sequential_pca.fit(X);

    // Observations:
    // (5,6), (6,7), (7,8)
    std::vector<double> X_new{
        5.0, 6.0, 7.0,
        6.0, 7.0, 8.0
    };

    batch_pca.batch_update(X_new);

    std::vector<double> x1{5.0, 6.0};
    std::vector<double> x2{6.0, 7.0};
    std::vector<double> x3{7.0, 8.0};

    sequential_pca.update(x1);
    sequential_pca.update(x2);
    sequential_pca.update(x3);

    EXPECT_EQ(batch_pca.n_observations(), 7);

    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(
            batch_pca.mean()[i],
            sequential_pca.mean()[i],
            tol);
    }

    EXPECT_NEAR(
        batch_pca.singular_values()[0],
        sequential_pca.singular_values()[0],
        tol);

    EXPECT_NEAR(
        batch_pca.explained_variance()[0],
        sequential_pca.explained_variance()[0],
        tol);

    // PCA directions have arbitrary sign.
    double dot = 0.0;

    for (std::size_t i = 0; i < 2; ++i) {
        dot += batch_pca.components()[i]
             * sequential_pca.components()[i];
    }

    EXPECT_NEAR(std::abs(dot), 1.0, tol);
}



TEST(PCATest, Reset)
{
    hdos::PCA pca(2, 1);

    std::vector<double> X{
        1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0
    };

    pca.fit(X);
    pca.reset();

    EXPECT_EQ(pca.n_observations(), 0);
    EXPECT_EQ(pca.n_features(), 2);
    EXPECT_EQ(pca.n_components(), 1);

    for (double x : pca.mean())
        EXPECT_DOUBLE_EQ(x, 0.0);

    for (double x : pca.components())
        EXPECT_DOUBLE_EQ(x, 0.0);

    for (double x : pca.singular_values())
        EXPECT_DOUBLE_EQ(x, 0.0);

    for (double x : pca.explained_variance())
        EXPECT_DOUBLE_EQ(x, 0.0);
}


TEST(PCATest, UpdateAfterResetThrows)
{
    hdos::PCA pca(2, 1);

    std::vector<double> X{
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0
    };

    pca.fit(X);
    pca.reset();

    std::vector<double> x{4.0, 4.0};

    EXPECT_THROW(
        pca.update(x),
        std::logic_error);
}