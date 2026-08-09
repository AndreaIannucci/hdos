#include <gtest/gtest.h>

#include <cmath>
#include <vector>
#include <random>


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



TEST(PCATest, FullRankUpdatesMatchBatchPCA)
{
    hdos::PCA incremental(2, 2);
    hdos::PCA batch(2, 2);

    // Initial observations:
    // (1,2), (2,-1), (4,3), (7,0)
    std::vector<double> X_initial{
        1.0, 2.0, 4.0, 7.0,
        2.0, -1.0, 3.0, 0.0
    };

    incremental.fit(X_initial);

    std::vector<double> x1{3.0, 4.0};
    std::vector<double> x2{8.0, -2.0};
    std::vector<double> x3{5.0, 1.0};

    incremental.update(x1);
    incremental.update(x2);
    incremental.update(x3);

    // Same seven observations all at once.
    std::vector<double> X_all{
        1.0, 2.0, 4.0, 7.0, 3.0, 8.0, 5.0,
        2.0, -1.0, 3.0, 0.0, 4.0, -2.0, 1.0
    };

    batch.fit(X_all);

    EXPECT_EQ(incremental.n_observations(), 7);

    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(
            incremental.mean()[i],
            batch.mean()[i],
            tol);

        EXPECT_NEAR(
            incremental.singular_values()[i],
            batch.singular_values()[i],
            tol);

        EXPECT_NEAR(
            incremental.explained_variance()[i],
            batch.explained_variance()[i],
            tol);
    }

    // Corresponding PCs should agree up to sign.
    for (std::size_t j = 0; j < 2; ++j) {

        double dot = 0.0;

        for (std::size_t i = 0; i < 2; ++i) {
            dot +=
                incremental.components()[i + j * 2] *
                batch.components()[i + j * 2];
        }

        EXPECT_NEAR(std::abs(dot), 1.0, tol);
    }
}



TEST(PCATest, ComponentsRemainOrthonormalAfterUpdates)
{
    hdos::PCA pca(3, 2);

    std::vector<double> X{
        1.0, 2.0, 3.0, 4.0, 5.0,
        0.0, 1.0, -1.0, 2.0, 0.0,
        2.0, -1.0, 0.0, 1.0, -2.0
    };

    pca.fit(X);

    std::vector<std::vector<double>> observations{
        {6.0,  1.0,  3.0},
        {0.0, -2.0,  1.0},
        {7.0,  3.0, -1.0},
        {-1.0, 1.0,  2.0}
    };

    for (const auto& x : observations) {
        pca.update(x);

        const auto V = pca.components();

        for (std::size_t j = 0; j < 2; ++j) {

            double norm_sq = 0.0;

            for (std::size_t i = 0; i < 3; ++i) {
                const double v = V[i + j * 3];
                norm_sq += v * v;
            }

            EXPECT_NEAR(norm_sq, 1.0, 1e-8);
        }

        double dot = 0.0;

        for (std::size_t i = 0; i < 3; ++i) {
            dot += V[i] * V[i + 3];
        }

        EXPECT_NEAR(dot, 0.0, 1e-8);
    }
}


TEST(PCATest, ConstantDataRemainsZeroVariance)
{
    hdos::PCA pca(3, 2);

    std::vector<double> X{
         3.0,  3.0,  3.0,  3.0,
        -2.0, -2.0, -2.0, -2.0,
         7.0,  7.0,  7.0,  7.0
    };

    pca.fit(X);

    std::vector<double> x{3.0, -2.0, 7.0};
    pca.update(x);

    EXPECT_EQ(pca.n_observations(), 5);

    EXPECT_NEAR(pca.mean()[0],  3.0, tol);
    EXPECT_NEAR(pca.mean()[1], -2.0, tol);
    EXPECT_NEAR(pca.mean()[2],  7.0, tol);

    for (double sigma : pca.singular_values()) {
        EXPECT_NEAR(sigma, 0.0, tol);
        EXPECT_TRUE(std::isfinite(sigma));
    }

    for (double var : pca.explained_variance()) {
        EXPECT_NEAR(var, 0.0, tol);
        EXPECT_TRUE(std::isfinite(var));
    }

    for (double v : pca.components()) {
        EXPECT_TRUE(std::isfinite(v));
    }
}



TEST(PCATest, TranslationDoesNotChangePrincipalComponents)
{
    hdos::PCA pca1(2, 1);
    hdos::PCA pca2(2, 1);

    std::vector<double> X{
        1.0, 2.0, 4.0, 5.0, 8.0,
       -1.0, 3.0, 0.0, 2.0, 7.0
    };

    // First coordinate +100, second coordinate -50.
    std::vector<double> X_shifted{
        101.0, 102.0, 104.0, 105.0, 108.0,
        -51.0, -47.0, -50.0, -48.0, -43.0
    };

    pca1.fit(X);
    pca2.fit(X_shifted);

    EXPECT_NEAR(
        pca1.singular_values()[0],
        pca2.singular_values()[0],
        tol);

    EXPECT_NEAR(
        pca1.explained_variance()[0],
        pca2.explained_variance()[0],
        tol);

    double dot = 0.0;

    for (std::size_t i = 0; i < 2; ++i) {
        dot += pca1.components()[i]
             * pca2.components()[i];
    }

    EXPECT_NEAR(std::abs(dot), 1.0, tol);
}


TEST(PCATest, RefitReplacesPreviousState)
{
    hdos::PCA pca(2, 1);

    std::vector<double> X1{
        1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0
    };

    pca.fit(X1);

    std::vector<double> x{5.0, 5.0};
    pca.update(x);

    EXPECT_EQ(pca.n_observations(), 5);

    // Three completely new observations.
    std::vector<double> X2{
        10.0, 20.0, 30.0,
         2.0,  4.0,  6.0
    };

    pca.fit(X2);

    EXPECT_EQ(pca.n_observations(), 3);
    EXPECT_NEAR(pca.mean()[0], 20.0, tol);
    EXPECT_NEAR(pca.mean()[1], 4.0, tol);
}




TEST(PCATest, ResetThenRefit)
{
    hdos::PCA pca(2, 1);

    std::vector<double> X1{
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0
    };

    pca.fit(X1);
    pca.reset();

    std::vector<double> X2{
        10.0, 20.0, 30.0, 40.0,
         0.0,  1.0,  2.0,  3.0
    };

    pca.fit(X2);

    EXPECT_EQ(pca.n_observations(), 4);
    EXPECT_NEAR(pca.mean()[0], 25.0, tol);
    EXPECT_NEAR(pca.mean()[1], 1.5, tol);
}



TEST(PCATest, UpdateRejectsWrongDimension)
{
    hdos::PCA pca(2, 1);

    std::vector<double> X{
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0
    };

    pca.fit(X);

    std::vector<double> bad_x{1.0, 2.0, 3.0};

    EXPECT_THROW(
        pca.update(bad_x),
        std::invalid_argument);

    EXPECT_EQ(pca.n_observations(), 3);
}


TEST(PCATest, UpdateBeforeFitThrows)
{
    hdos::PCA pca(2, 1);

    std::vector<double> x{1.0, 2.0};

    EXPECT_THROW(
        pca.update(x),
        std::logic_error);

    EXPECT_EQ(pca.n_observations(), 0);
}




TEST(PCATest, UpdateRejectsNaN)
{
    hdos::PCA pca(2, 1);

    std::vector<double> X{
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0
    };

    pca.fit(X);

    std::vector<double> bad_x{
        std::numeric_limits<double>::quiet_NaN(),
        4.0
    };

    EXPECT_THROW(
        pca.update(bad_x),
        std::invalid_argument);

    // Invalid update must not count as an observation.
    EXPECT_EQ(pca.n_observations(), 3);
}



TEST(PCATest, UpdateRejectsInfinity)
{
    hdos::PCA pca(2, 1);

    std::vector<double> X{
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0
    };

    pca.fit(X);

    std::vector<double> bad_x{
        std::numeric_limits<double>::infinity(),
        4.0
    };

    EXPECT_THROW(
        pca.update(bad_x),
        std::invalid_argument);

    EXPECT_EQ(pca.n_observations(), 3);
}




TEST(PCATest, BatchUpdateRejectsInvalidDimensions)
{
    hdos::PCA pca(2, 1);

    std::vector<double> X{
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0
    };

    pca.fit(X);

    std::vector<double> bad_batch{
        1.0, 2.0, 3.0
    };

    EXPECT_THROW(
        pca.batch_update(bad_batch),
        std::invalid_argument);

    EXPECT_EQ(pca.n_observations(), 3);
}




TEST(PCATest, EmptyBatchDoesNothing)
{
    hdos::PCA pca(2, 1);

    std::vector<double> X{
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0
    };

    pca.fit(X);

    const double mean0 = pca.mean()[0];
    const double mean1 = pca.mean()[1];
    const double sigma = pca.singular_values()[0];

    std::vector<double> empty;

    pca.batch_update(empty);

    EXPECT_EQ(pca.n_observations(), 3);
    EXPECT_NEAR(pca.mean()[0], mean0, tol);
    EXPECT_NEAR(pca.mean()[1], mean1, tol);
    EXPECT_NEAR(pca.singular_values()[0], sigma, tol);
}




TEST(PCATest, RandomFullRankUpdatesMatchBatchPCA)
{
    constexpr std::size_t d = 3;
    constexpr std::size_t n_initial = 10;
    constexpr std::size_t n_updates = 25;

    std::mt19937 rng(123456);
    std::normal_distribution<double> dist(0.0, 3.0);

    std::vector<std::vector<double>> observations;

    for (std::size_t i = 0; i < n_initial; ++i) {
        std::vector<double> x(d);

        for (double& value : x) {
            value = dist(rng);
        }

        observations.push_back(x);
    }

    // Construct initial matrix in column-major order.
    std::vector<double> X_initial(n_initial * d);

    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t i = 0; i < n_initial; ++i) {
            X_initial[i + j * n_initial] =
                observations[i][j];
        }
    }

    hdos::PCA incremental(d, d);
    incremental.fit(X_initial);

    // Add observations incrementally.
    for (std::size_t k = 0; k < n_updates; ++k) {
        std::vector<double> x(d);

        for (double& value : x) {
            value = dist(rng);
        }

        incremental.update(x);
        observations.push_back(x);
    }

    const std::size_t n_total = observations.size();

    // Construct complete dataset.
    std::vector<double> X_all(n_total * d);

    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t i = 0; i < n_total; ++i) {
            X_all[i + j * n_total] =
                observations[i][j];
        }
    }

    hdos::PCA batch(d, d);
    batch.fit(X_all);

    EXPECT_EQ(incremental.n_observations(), n_total);

    // Means.
    for (std::size_t j = 0; j < d; ++j) {
        EXPECT_NEAR(
            incremental.mean()[j],
            batch.mean()[j],
            1e-9);
    }

    // Singular values.
    for (std::size_t j = 0; j < d; ++j) {
        EXPECT_NEAR(
            incremental.singular_values()[j],
            batch.singular_values()[j],
            1e-8);
    }

    // Reconstruct V Sigma^2 V^T.
    std::vector<double> S_inc(d * d, 0.0);
    std::vector<double> S_batch(d * d, 0.0);

    for (std::size_t j = 0; j < d; ++j) {
        for (std::size_t i = 0; i < d; ++i) {

            for (std::size_t k = 0; k < d; ++k) {

                const double sigma_inc =
                    incremental.singular_values()[k];

                const double sigma_batch =
                    batch.singular_values()[k];

                S_inc[i + j * d] +=
                    incremental.components()[i + k * d] *
                    sigma_inc * sigma_inc *
                    incremental.components()[j + k * d];

                S_batch[i + j * d] +=
                    batch.components()[i + k * d] *
                    sigma_batch * sigma_batch *
                    batch.components()[j + k * d];
            }
        }
    }

    for (std::size_t k = 0; k < d * d; ++k) {
        const double comparison_tol =
            1e-8 * std::max(1.0, std::abs(S_batch[k]));

        EXPECT_NEAR(
            S_inc[k],
            S_batch[k],
            comparison_tol);
    }
}


TEST(PCATest, RandomUpdatesPreservePCAInvariants)
{
    constexpr std::size_t d = 5;
    constexpr std::size_t k = 2;
    constexpr std::size_t n_initial = 20;

    std::mt19937 rng(654321);
    std::normal_distribution<double> dist(0.0, 2.0);

    std::vector<double> X(n_initial * d);

    for (double& value : X) {
        value = dist(rng);
    }

    hdos::PCA pca(d, k);
    pca.fit(X);

    for (std::size_t step = 0; step < 100; ++step) {

        std::vector<double> x(d);

        for (double& value : x) {
            value = dist(rng);
        }

        pca.update(x);

        const auto V = pca.components();
        const auto sigma = pca.singular_values();
        const auto variance = pca.explained_variance();

        // Singular values stay ordered and finite.
        EXPECT_TRUE(std::isfinite(sigma[0]));
        EXPECT_TRUE(std::isfinite(sigma[1]));

        EXPECT_GE(sigma[0], sigma[1]);
        EXPECT_GE(sigma[1], 0.0);

        // Explained variance remains consistent.
        const double denominator =
            static_cast<double>(
                pca.n_observations() - 1);

        for (std::size_t j = 0; j < k; ++j) {
            EXPECT_NEAR(
                variance[j],
                sigma[j] * sigma[j] / denominator,
                1e-9);
        }

        // Components have unit norm.
        for (std::size_t j = 0; j < k; ++j) {

            double norm_sq = 0.0;

            for (std::size_t i = 0; i < d; ++i) {
                const double value = V[i + j * d];

                EXPECT_TRUE(std::isfinite(value));

                norm_sq += value * value;
            }

            EXPECT_NEAR(norm_sq, 1.0, 1e-8);
        }

        // Components are mutually orthogonal.
        double dot = 0.0;

        for (std::size_t i = 0; i < d; ++i) {
            dot += V[i] * V[i + d];
        }

        EXPECT_NEAR(dot, 0.0, 1e-8);
    }
}