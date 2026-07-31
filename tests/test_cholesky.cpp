#include <gtest/gtest.h>

#include "detail/cholesky.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void expect_factorizes(
    const std::vector<double>& lower,
    const std::vector<double>& matrix,
    const std::size_t n,
    const double tolerance = 1e-12)
{
    ASSERT_EQ(lower.size(), n * n);
    ASSERT_EQ(matrix.size(), n * n);

    // Verify L * L^T = matrix.
    for (std::size_t column = 0; column < n; ++column) {
        for (std::size_t row = 0; row < n; ++row) {
            double reconstructed = 0.0;

            const std::size_t last =
                std::min(row, column);

            for (std::size_t k = 0; k <= last; ++k) {
                reconstructed +=
                    lower[row + k * n] *
                    lower[column + k * n];
            }

            EXPECT_NEAR(
                reconstructed,
                matrix[row + column * n],
                tolerance
            );
        }
    }

    // Verify that L is lower triangular.
    for (std::size_t column = 1; column < n; ++column) {
        for (std::size_t row = 0; row < column; ++row) {
            EXPECT_NEAR(
                lower[row + column * n],
                0.0,
                tolerance
            );
        }
    }

    // Verify that the diagonal is positive.
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_GT(lower[i + i * n], 0.0);
    }
}

} // namespace

TEST(CholeskyDecomposition, DecomposesPositiveDefiniteMatrix)
{
    // Column-major representation of:
    //
    // [4  2]
    // [2  3]
    const std::vector<double> matrix{
        4.0, 2.0,
        2.0, 3.0
    };

    const std::vector<double> lower =
        hdos::detail::cholesky_decomp(matrix);

    ASSERT_EQ(lower.size(), 4U);

    EXPECT_NEAR(lower[0], 2.0, 1e-12);
    EXPECT_NEAR(lower[1], 1.0, 1e-12);
    EXPECT_NEAR(lower[2], 0.0, 1e-12);
    EXPECT_NEAR(lower[3], std::sqrt(2.0), 1e-12);

    expect_factorizes(lower, matrix, 2);
}

TEST(CholeskyDecomposition, DecomposesThreeByThreeMatrix)
{
    // Matrix:
    //
    // [25  15  -5]
    // [15  18   0]
    // [-5   0  11]
    //
    // Its Cholesky factor is:
    //
    // [ 5  0  0]
    // [ 3  3  0]
    // [-1  1  3]
    const std::vector<double> matrix{
        25.0, 15.0, -5.0,
        15.0, 18.0,  0.0,
        -5.0,  0.0, 11.0
    };

    const std::vector<double> expected{
         5.0, 3.0, -1.0,
         0.0, 3.0,  1.0,
         0.0, 0.0,  3.0
    };

    const std::vector<double> lower =
        hdos::detail::cholesky_decomp(matrix);

    ASSERT_EQ(lower.size(), expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(lower[i], expected[i], 1e-12);
    }

    expect_factorizes(lower, matrix, 3);
}

TEST(CholeskyDecomposition, DoesNotModifyInput)
{
    std::vector<double> matrix{
        4.0, 2.0,
        2.0, 3.0
    };

    const std::vector<double> original = matrix;

    const std::vector<double> lower =
        hdos::detail::cholesky_decomp(matrix);

    EXPECT_EQ(matrix, original);
    expect_factorizes(lower, original, 2);
}

TEST(CholeskyDecomposition, RejectsNonSquareMatrix)
{
    const std::vector<double> matrix{
        1.0, 2.0, 3.0
    };

    EXPECT_THROW(
        hdos::detail::cholesky_decomp(matrix),
        std::invalid_argument
    );
}

TEST(CholeskyDecomposition, RejectsIndefiniteMatrix)
{
    const std::vector<double> matrix{
        1.0, 2.0,
        2.0, 1.0
    };

    EXPECT_THROW(
        hdos::detail::cholesky_decomp(matrix),
        std::invalid_argument
    );
}

TEST(CholeskyDecomposition, RejectsSingularMatrix)
{
    const std::vector<double> matrix{
        1.0, 1.0,
        1.0, 1.0
    };

    EXPECT_THROW(
        hdos::detail::cholesky_decomp(matrix),
        std::invalid_argument
    );
}

TEST(RankOneCholeskyUpdate, UpdatesWithoutIntercept)
{
    // Original matrix:
    //
    // [4  2]
    // [2  3]
    //
    // Add 0.5 * x*x^T, where x = [1, 2]:
    //
    // [4.5  3]
    // [3    5]
    const std::vector<double> matrix{
        4.0, 2.0,
        2.0, 3.0
    };

    const std::vector<double> expected{
        4.5, 3.0,
        3.0, 5.0
    };

    const std::vector<double> x{1.0, 2.0};
    const std::vector<double> original_x = x;

    std::vector<double> lower =
        hdos::detail::cholesky_decomp(matrix);

    hdos::detail::rk1_cholesky(
        lower,
        x,
        0.5
    );

    expect_factorizes(lower, expected, 2);
    EXPECT_EQ(x, original_x);
}

TEST(RankOneCholeskyUpdate, UpdatesWithIntercept)
{
    // Start from I_3 and use x = [2, -1].
    //
    // The logical update vector is:
    //
    // z = [2, -1, 1].
    //
    // Therefore I_3 + z*z^T is:
    //
    // [ 5  -2   2]
    // [-2   2  -1]
    // [ 2  -1   2]
    const std::vector<double> identity{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };

    const std::vector<double> expected{
         5.0, -2.0,  2.0,
        -2.0,  2.0, -1.0,
         2.0, -1.0,  2.0
    };

    const std::vector<double> x{2.0, -1.0};
    const std::vector<double> original_x = x;

    std::vector<double> lower =
        hdos::detail::cholesky_decomp(identity);

    hdos::detail::rk1_cholesky(
        lower,
        x,
        1.0,
        true
    );

    expect_factorizes(lower, expected, 3);
    EXPECT_EQ(x, original_x);
}

TEST(RankOneCholeskyUpdate, SupportsRepeatedUpdates)
{
    const std::vector<double> identity{
        1.0, 0.0,
        0.0, 1.0
    };

    const std::vector<double> x1{1.0, 2.0};
    const std::vector<double> x2{-1.0, 0.5};

    // I + x1*x1^T + 0.5*x2*x2^T
    const std::vector<double> expected{
        2.5,  1.75,
        1.75, 5.125
    };

    std::vector<double> lower =
        hdos::detail::cholesky_decomp(identity);

    hdos::detail::rk1_cholesky(lower, x1);
    hdos::detail::rk1_cholesky(lower, x2, 0.5);

    expect_factorizes(lower, expected, 2);
}