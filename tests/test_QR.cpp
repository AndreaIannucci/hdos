#include <gtest/gtest.h>

#include "detail/QR.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

TEST(QRDecomposition, ProducesExpectedPackedFactorisation)
{
    // Column-major representation of:
    //
    // [4  1]
    // [3  2]
    // [0  2]
    std::vector<double> matrix{
        4.0, 3.0, 0.0,
        1.0, 2.0, 2.0
    };

    const std::vector<double> tau =
        hdos::detail::QR_decomp(
            matrix,
            3,
            2
        );

    // First reflector:
    //
    // [4, 3, 0]^T -> [-5, 0, 0]^T
    //
    // beta   = -5
    // v_tail = [1/3, 0]
    // tau    = 9/5
    //
    // After applying it, the second column becomes:
    //
    // [-2, 1, 2]^T
    //
    // The second reflector acts on [1, 2]^T:
    //
    // beta   = -sqrt(5)
    // v_tail = 2 / (1 + sqrt(5))
    // tau    = 1 + 1/sqrt(5)

    ASSERT_EQ(matrix.size(), 6U);
    ASSERT_EQ(tau.size(), 2U);

    const double sqrt_five = std::sqrt(5.0);

    // First packed column.
    EXPECT_NEAR(matrix[0], -5.0, 1e-12);
    EXPECT_NEAR(matrix[1], 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(matrix[2], 0.0, 1e-12);

    // Second packed column.
    EXPECT_NEAR(matrix[3], -2.0, 1e-12);
    EXPECT_NEAR(matrix[4], -sqrt_five, 1e-12);
    EXPECT_NEAR(
        matrix[5],
        2.0 / (1.0 + sqrt_five),
        1e-12
    );

    EXPECT_NEAR(tau[0], 9.0 / 5.0, 1e-12);
    EXPECT_NEAR(
        tau[1],
        1.0 + 1.0 / sqrt_five,
        1e-12
    );
}

TEST(QRDecomposition, LeavesUpperTriangularMatrixPackedAsItIs)
{
    // Column-major representation of:
    //
    // [2  1  4]
    // [0  3  5]
    // [0  0  6]
    //
    // Every column tail is already zero.
    const std::vector<double> original{
        2.0, 0.0, 0.0,
        1.0, 3.0, 0.0,
        4.0, 5.0, 6.0
    };

    std::vector<double> matrix = original;

    const std::vector<double> tau =
        hdos::detail::QR_decomp(
            matrix,
            3,
            3
        );

    ASSERT_EQ(tau.size(), 3U);
    EXPECT_EQ(matrix, original);

    EXPECT_EQ(tau[0], 0.0);
    EXPECT_EQ(tau[1], 0.0);
    EXPECT_EQ(tau[2], 0.0);
}

TEST(QRDecomposition, HandlesWideMatrix)
{
    // Column-major representation of:
    //
    // [1  2  3]
    // [0  4  5]
    std::vector<double> matrix{
        1.0, 0.0,
        2.0, 4.0,
        3.0, 5.0
    };

    const std::vector<double> tau =
        hdos::detail::QR_decomp(
            matrix,
            2,
            3
        );

    // min(rows, cols) = 2.
    ASSERT_EQ(tau.size(), 2U);

    // This matrix is already upper trapezoidal.
    EXPECT_EQ(matrix[0], 1.0);
    EXPECT_EQ(matrix[1], 0.0);

    EXPECT_EQ(matrix[2], 2.0);
    EXPECT_EQ(matrix[3], 4.0);

    EXPECT_EQ(matrix[4], 3.0);
    EXPECT_EQ(matrix[5], 5.0);

    EXPECT_EQ(tau[0], 0.0);
    EXPECT_EQ(tau[1], 0.0);
}

TEST(QRDecomposition, HandlesSingleColumn)
{
    // [3, 4]^T should be transformed into [beta, v_tail].
    std::vector<double> matrix{
        3.0,
        4.0
    };

    const std::vector<double> tau =
        hdos::detail::QR_decomp(
            matrix,
            2,
            1
        );

    ASSERT_EQ(tau.size(), 1U);

    EXPECT_NEAR(matrix[0], -5.0, 1e-12);
    EXPECT_NEAR(matrix[1], 0.5, 1e-12);
    EXPECT_NEAR(tau[0], 1.6, 1e-12);
}

TEST(QRDecomposition, RejectsIncompatibleMatrixDimensions)
{
    // A 3 x 2 matrix requires six entries.
    std::vector<double> matrix{
        1.0, 2.0, 3.0, 4.0, 5.0
    };

    EXPECT_THROW(
        hdos::detail::QR_decomp(
            matrix,
            3,
            2
        ),
        std::invalid_argument
    );
}

TEST(QRDecomposition, RejectsZeroRows)
{
    std::vector<double> matrix;

    EXPECT_THROW(
        hdos::detail::QR_decomp(
            matrix,
            0,
            3
        ),
        std::invalid_argument
    );
}

TEST(QRDecomposition, RejectsZeroColumns)
{
    std::vector<double> matrix;

    EXPECT_THROW(
        hdos::detail::QR_decomp(
            matrix,
            3,
            0
        ),
        std::invalid_argument
    );
}