#include <gtest/gtest.h>

#include "detail/householder.hpp"

#include <stdexcept>
#include <vector>

TEST(HouseholderRight, AppliesReflectorToEntireMatrix)
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
    // [1  2]
    // [3  4]
    // [5  6]
    std::vector<double> matrix{
        1.0, 3.0, 5.0,
        2.0, 4.0, 6.0
    };

    hdos::detail::apply_householder_right(
        matrix,
        3,
        2,
        0,
        0,
        packed_reflector,
        tau
    );

    // A * H =
    //
    // [-2.0   1.0]
    // [-4.8   1.4]
    // [-7.6   1.8]
    ASSERT_EQ(matrix.size(), 6U);

    EXPECT_NEAR(matrix[0], -2.0, 1e-12);
    EXPECT_NEAR(matrix[1], -4.8, 1e-12);
    EXPECT_NEAR(matrix[2], -7.6, 1e-12);

    EXPECT_NEAR(matrix[3], 1.0, 1e-12);
    EXPECT_NEAR(matrix[4], 1.4, 1e-12);
    EXPECT_NEAR(matrix[5], 1.8, 1e-12);
}

TEST(HouseholderRight, AppliesReflectorToTrailingBlock)
{
    // Column-major representation of:
    //
    // [1  2  3]
    // [4  4  3]
    // [6  5  7]
    //
    // Apply the reflector only to rows 1:3 and columns 1:3.
    //
    // The affected block is:
    //
    // [4  3]
    // [5  7]
    std::vector<double> matrix{
        1.0, 4.0, 6.0,
        2.0, 4.0, 5.0,
        3.0, 3.0, 7.0
    };

    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0
    };

    hdos::detail::apply_householder_right(
        matrix,
        3,
        3,
        1,
        1,
        packed_reflector,
        9.0 / 5.0
    );

    // The first row and first column are unchanged.
    //
    // The trailing block becomes:
    //
    // [-5.0   0.0]
    // [-8.2   2.6]
    ASSERT_EQ(matrix.size(), 9U);

    EXPECT_NEAR(matrix[0], 1.0, 1e-12);
    EXPECT_NEAR(matrix[1], 4.0, 1e-12);
    EXPECT_NEAR(matrix[2], 6.0, 1e-12);

    EXPECT_NEAR(matrix[3],  2.0, 1e-12);
    EXPECT_NEAR(matrix[4], -5.0, 1e-12);
    EXPECT_NEAR(matrix[5], -8.2, 1e-12);

    EXPECT_NEAR(matrix[6], 3.0, 1e-12);
    EXPECT_NEAR(matrix[7], 0.0, 1e-12);
    EXPECT_NEAR(matrix[8], 2.6, 1e-12);
}

TEST(HouseholderRight, UsesImplicitFirstReflectorEntry)
{
    // packed_reflector[0] stores beta, not v[0].
    // The actual reflector vector is v = [1, 1/3]^T.
    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0
    };

    // A 1 x 2 matrix.
    std::vector<double> matrix{
        4.0,
        3.0
    };

    hdos::detail::apply_householder_right(
        matrix,
        1,
        2,
        0,
        0,
        packed_reflector,
        9.0 / 5.0
    );

    // [4, 3]H = [-5, 0].
    ASSERT_EQ(matrix.size(), 2U);
    EXPECT_NEAR(matrix[0], -5.0, 1e-12);
    EXPECT_NEAR(matrix[1], 0.0, 1e-12);
}

TEST(HouseholderRight, DoesNothingWhenTauIsZero)
{
    const std::vector<double> packed_reflector{
        5.0,
        0.0
    };

    const std::vector<double> original{
        1.0, 3.0, 5.0,
        2.0, 4.0, 6.0
    };

    std::vector<double> matrix = original;

    hdos::detail::apply_householder_right(
        matrix,
        3,
        2,
        0,
        0,
        packed_reflector,
        0.0
    );

    EXPECT_EQ(matrix, original);
}

TEST(HouseholderRight, DoesNothingForEmptyRowBlock)
{
    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0
    };

    const std::vector<double> original{
        1.0, 3.0,
        2.0, 4.0
    };

    std::vector<double> matrix = original;

    hdos::detail::apply_householder_right(
        matrix,
        2,
        2,
        2,
        0,
        packed_reflector,
        9.0 / 5.0
    );

    EXPECT_EQ(matrix, original);
}

TEST(HouseholderRight, DoesNothingForEmptyColumnBlock)
{
    const std::vector<double> original{
        1.0, 3.0,
        2.0, 4.0
    };

    std::vector<double> matrix = original;
    const std::vector<double> packed_reflector;

    hdos::detail::apply_householder_right(
        matrix,
        2,
        2,
        0,
        2,
        packed_reflector,
        1.0
    );

    EXPECT_EQ(matrix, original);
}

TEST(HouseholderRight, RejectsIncompatibleReflectorDimension)
{
    std::vector<double> matrix{
        1.0, 3.0,
        2.0, 4.0
    };

    // start_col = 0 means the reflector must have two entries.
    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0,
        0.0
    };

    EXPECT_THROW(
        hdos::detail::apply_householder_right(
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

TEST(HouseholderRight, RejectsReflectorIncompatibleWithStartColumn)
{
    std::vector<double> matrix{
        1.0, 4.0,
        2.0, 5.0,
        3.0, 6.0
    };

    // cols = 3 and start_col = 1, so the reflector
    // must have exactly two entries.
    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0,
        0.0
    };

    EXPECT_THROW(
        hdos::detail::apply_householder_right(
            matrix,
            2,
            3,
            0,
            1,
            packed_reflector,
            9.0 / 5.0
        ),
        std::invalid_argument
    );
}

TEST(HouseholderRight, RejectsIncompatibleMatrixDimensions)
{
    // A 3 x 2 matrix requires six elements.
    std::vector<double> matrix{
        1.0, 3.0, 5.0,
        2.0, 4.0
    };

    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0
    };

    EXPECT_THROW(
        hdos::detail::apply_householder_right(
            matrix,
            3,
            2,
            0,
            0,
            packed_reflector,
            9.0 / 5.0
        ),
        std::invalid_argument
    );
}

TEST(HouseholderRight, RejectsInvalidStartingRow)
{
    std::vector<double> matrix{
        1.0, 3.0,
        2.0, 4.0
    };

    const std::vector<double> packed_reflector{
        -5.0,
        1.0 / 3.0
    };

    EXPECT_THROW(
        hdos::detail::apply_householder_right(
            matrix,
            2,
            2,
            3,
            0,
            packed_reflector,
            9.0 / 5.0
        ),
        std::invalid_argument
    );
}

TEST(HouseholderRight, RejectsInvalidStartingColumn)
{
    std::vector<double> matrix{
        1.0, 3.0,
        2.0, 4.0
    };

    const std::vector<double> packed_reflector;

    EXPECT_THROW(
        hdos::detail::apply_householder_right(
            matrix,
            2,
            2,
            0,
            3,
            packed_reflector,
            1.0
        ),
        std::invalid_argument
    );
}