#include <gtest/gtest.h>

#include "detail/householder.hpp"

#include <stdexcept>
#include <vector>

TEST(HouseholderLeft, AppliesReflectorToEntireMatrix)
{
    // Reflector generated from x = [4, 3]^T:
    //
    // beta = -5
    // v    = [1, 1/3]^T
    // tau  = 9/5
    //
    // H = [-4/5  -3/5]
    //     [-3/5   4/5]
    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0
    };

    const double tau = 9.0 / 5.0;

    // Column-major representation of:
    //
    // [1  2  3]
    // [4  5  6]
    std::vector<double> matrix{
        1.0, 4.0,
        2.0, 5.0,
        3.0, 6.0
    };

    hdos::detail::apply_householder_left(
        matrix,
        2,
        3,
        0,
        0,
        packed_reflector,
        tau
    );

    // H * A =
    //
    // [-3.2  -4.6  -6.0]
    // [ 2.6   2.8   3.0]
    ASSERT_EQ(matrix.size(), 6U);

    EXPECT_NEAR(matrix[0], -3.2, 1e-12);
    EXPECT_NEAR(matrix[1],  2.6, 1e-12);

    EXPECT_NEAR(matrix[2], -4.6, 1e-12);
    EXPECT_NEAR(matrix[3],  2.8, 1e-12);

    EXPECT_NEAR(matrix[4], -6.0, 1e-12);
    EXPECT_NEAR(matrix[5],  3.0, 1e-12);
}

TEST(HouseholderLeft, AppliesReflectorToTrailingBlock)
{
    // Column-major representation of:
    //
    // [1  2  3]
    // [4  4  5]
    // [6  3  7]
    //
    // Apply the reflector only to rows 1:3 and columns 1:3.
    //
    // The affected block is:
    //
    // [4  5]
    // [3  7]
    std::vector<double> matrix{
        1.0, 4.0, 6.0,
        2.0, 4.0, 3.0,
        3.0, 5.0, 7.0
    };

    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0
    };

    hdos::detail::apply_householder_left(
        matrix,
        3,
        3,
        1,
        1,
        packed_reflector,
        9.0 / 5.0
    );

    // The unaffected first row and first column remain unchanged.
    //
    // The trailing block becomes:
    //
    // [-5.0  -8.2]
    // [ 0.0   2.6]
    ASSERT_EQ(matrix.size(), 9U);

    EXPECT_NEAR(matrix[0], 1.0, 1e-12);
    EXPECT_NEAR(matrix[1], 4.0, 1e-12);
    EXPECT_NEAR(matrix[2], 6.0, 1e-12);

    EXPECT_NEAR(matrix[3],  2.0, 1e-12);
    EXPECT_NEAR(matrix[4], -5.0, 1e-12);
    EXPECT_NEAR(matrix[5],  0.0, 1e-12);

    EXPECT_NEAR(matrix[6],  3.0, 1e-12);
    EXPECT_NEAR(matrix[7], -8.2, 1e-12);
    EXPECT_NEAR(matrix[8],  2.6, 1e-12);
}

TEST(HouseholderLeft, UsesImplicitFirstReflectorEntry)
{
    // packed_reflector[0] is beta, not v[0].
    // The actual reflector vector is v = [1, 1/3]^T.
    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0
    };

    std::vector<double> matrix{
        4.0,
        3.0
    };

    hdos::detail::apply_householder_left(
        matrix,
        2,
        1,
        0,
        0,
        packed_reflector,
        9.0 / 5.0
    );

    // The reflector maps [4, 3]^T to [-5, 0]^T.
    ASSERT_EQ(matrix.size(), 2U);
    EXPECT_NEAR(matrix[0], -5.0, 1e-12);
    EXPECT_NEAR(matrix[1],  0.0, 1e-12);
}

TEST(HouseholderLeft, DoesNothingWhenTauIsZero)
{
    const std::vector<double> packed_reflector{
        5.0,
        0.0
    };

    const std::vector<double> original{
        1.0, 4.0,
        2.0, 5.0,
        3.0, 6.0
    };

    std::vector<double> matrix = original;

    hdos::detail::apply_householder_left(
        matrix,
        2,
        3,
        0,
        0,
        packed_reflector,
        0.0
    );

    EXPECT_EQ(matrix, original);
}

TEST(HouseholderLeft, DoesNothingForEmptyColumnBlock)
{
    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0
    };

    const std::vector<double> original{
        1.0, 4.0,
        2.0, 5.0
    };

    std::vector<double> matrix = original;

    hdos::detail::apply_householder_left(
        matrix,
        2,
        2,
        0,
        2,
        packed_reflector,
        9.0 / 5.0
    );

    EXPECT_EQ(matrix, original);
}

TEST(HouseholderLeft, RejectsIncompatibleReflectorDimension)
{
    std::vector<double> matrix{
        1.0, 4.0,
        2.0, 5.0
    };

    // start_row = 0 means the reflector must have two entries.
    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0,
        0.0
    };

    EXPECT_THROW(
        hdos::detail::apply_householder_left(
            matrix,
            2,
            2,
            0,
            0,
            packed_reflector,
            9.0 / 5.0
        ),
        std::invalid_argument
    );
}

TEST(HouseholderLeft, RejectsReflectorIncompatibleWithStartRow)
{
    std::vector<double> matrix{
        1.0, 4.0, 6.0,
        2.0, 5.0, 7.0
    };

    // With rows = 3 and start_row = 1, the reflector
    // must have exactly two entries.
    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0,
        0.0
    };

    EXPECT_THROW(
        hdos::detail::apply_householder_left(
            matrix,
            3,
            2,
            1,
            0,
            packed_reflector,
            9.0 / 5.0
        ),
        std::invalid_argument
    );
}

TEST(HouseholderLeft, RejectsIncompatibleMatrixDimensions)
{
    // A 2 x 3 matrix needs six elements.
    std::vector<double> matrix{
        1.0, 2.0, 3.0, 4.0, 5.0
    };

    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0
    };

    EXPECT_THROW(
        hdos::detail::apply_householder_left(
            matrix,
            2,
            3,
            0,
            0,
            packed_reflector,
            9.0 / 5.0
        ),
        std::invalid_argument
    );
}

TEST(HouseholderLeft, RejectsInvalidStartingRow)
{
    std::vector<double> matrix{
        1.0, 4.0,
        2.0, 5.0
    };

    const std::vector<double> packed_reflector;

    EXPECT_THROW(
        hdos::detail::apply_householder_left(
            matrix,
            2,
            2,
            3,
            0,
            packed_reflector,
            1.0
        ),
        std::invalid_argument
    );
}

TEST(HouseholderLeft, RejectsInvalidStartingColumn)
{
    std::vector<double> matrix{
        1.0, 4.0,
        2.0, 5.0
    };

    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0
    };

    EXPECT_THROW(
        hdos::detail::apply_householder_left(
            matrix,
            2,
            2,
            0,
            3,
            packed_reflector,
            9.0 / 5.0
        ),
        std::invalid_argument
    );
}