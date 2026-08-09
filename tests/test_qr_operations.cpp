#include <gtest/gtest.h>

#include "detail/QR.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace {

void expect_valid_QR(
    const std::vector<double>& original,
    std::size_t rows,
    std::size_t cols,
    double tolerance = 1e-10)
{
    std::vector<double> packed_qr = original;

    const std::vector<double> tau =
        hdos::detail::QR_decomp(
            packed_qr,
            rows,
            cols
        );

    const std::vector<double> Q =
        hdos::detail::form_Q(
            packed_qr,
            tau,
            rows,
            cols
        );

    const std::vector<double> R =
        hdos::detail::extract_R(
            packed_qr,
            rows,
            cols
        );

    const std::size_t reduced_rows =
        std::min(rows, cols);

    ASSERT_EQ(
        Q.size(),
        rows * reduced_rows
    );

    ASSERT_EQ(
        R.size(),
        reduced_rows * cols
    );

    // Check Q^T Q = I.
    for (std::size_t j = 0; j < reduced_rows; ++j) {
        for (std::size_t k = 0; k < reduced_rows; ++k) {

            double dot = 0.0;

            for (std::size_t i = 0; i < rows; ++i) {
                dot +=
                    Q[i + j * rows] *
                    Q[i + k * rows];
            }

            const double expected =
                (j == k) ? 1.0 : 0.0;

            EXPECT_NEAR(
                dot,
                expected,
                tolerance
            );
        }
    }

    // Check Q R = A.
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {

            double value = 0.0;

            for (std::size_t k = 0;
                 k < reduced_rows;
                 ++k)
            {
                value +=
                    Q[i + k * rows] *
                    R[k + j * reduced_rows];
            }

            EXPECT_NEAR(
                value,
                original[i + j * rows],
                tolerance
            );
        }
    }
}

}

TEST(ExtractR, ExtractsReducedRFromTallPackedMatrix)
{
    // Packed column-major representation:
    //
    // [-5       -2        ]
    // [1/3      -sqrt(5)  ]
    // [0         v_tail   ]
    //
    // Entries below the diagonal store reflector tails.
    const double sqrt_five = std::sqrt(5.0);

    const std::vector<double> packed{
        -5.0, 1.0 / 3.0, 0.0,
        -2.0, -sqrt_five, 0.75
    };

    const std::vector<double> R =
        hdos::detail::extract_R(
            packed,
            3,
            2
        );

    // Reduced R has shape 2 x 2:
    //
    // [-5  -2       ]
    // [ 0  -sqrt(5) ]
    const std::vector<double> expected{
        -5.0, 0.0,
        -2.0, -sqrt_five
    };

    EXPECT_EQ(R, expected);
}

TEST(ExtractR, ExtractsReducedRFromSquarePackedMatrix)
{
    // Packed representation:
    //
    // [1  2  4]
    // [7  3  5]
    // [8  9  6]
    //
    // 7, 8 and 9 are packed reflector entries.
    const std::vector<double> packed{
        1.0, 7.0, 8.0,
        2.0, 3.0, 9.0,
        4.0, 5.0, 6.0
    };

    const std::vector<double> R =
        hdos::detail::extract_R(
            packed,
            3,
            3
        );

    const std::vector<double> expected{
        1.0, 0.0, 0.0,
        2.0, 3.0, 0.0,
        4.0, 5.0, 6.0
    };

    EXPECT_EQ(R, expected);
}

TEST(ExtractR, ExtractsReducedRFromWidePackedMatrix)
{
    // Packed 2 x 3 matrix:
    //
    // [2   3  5]
    // [99  4  6]
    //
    // 99 is the tail of the first reflector.
    const std::vector<double> packed{
        2.0, 99.0,
        3.0, 4.0,
        5.0, 6.0
    };

    const std::vector<double> R =
        hdos::detail::extract_R(
            packed,
            2,
            3
        );

    // Reduced R has shape 2 x 3.
    const std::vector<double> expected{
        2.0, 0.0,
        3.0, 4.0,
        5.0, 6.0
    };

    EXPECT_EQ(R, expected);
}

TEST(ExtractR, HandlesSingleColumn)
{
    const std::vector<double> packed{
        -5.0,
        0.5,
        0.25
    };

    const std::vector<double> R =
        hdos::detail::extract_R(
            packed,
            3,
            1
        );

    const std::vector<double> expected{
        -5.0
    };

    EXPECT_EQ(R, expected);
}

TEST(ExtractR, RejectsIncompatibleDimensions)
{
    const std::vector<double> packed{
        1.0, 2.0, 3.0
    };

    EXPECT_THROW(
        hdos::detail::extract_R(
            packed,
            2,
            2
        ),
        std::invalid_argument
    );
}

TEST(ExtractR, RejectsZeroDimensions)
{
    const std::vector<double> packed;

    EXPECT_THROW(
        hdos::detail::extract_R(
            packed,
            0,
            2
        ),
        std::invalid_argument
    );

    EXPECT_THROW(
        hdos::detail::extract_R(
            packed,
            2,
            0
        ),
        std::invalid_argument
    );
}


TEST(ApplyQTranspose, TransformsOriginalMatrixIntoR)
{
    // Column-major representation of:
    //
    // [4  1]
    // [3  2]
    // [0  2]
    const std::vector<double> original{
        4.0, 3.0, 0.0,
        1.0, 2.0, 2.0
    };

    std::vector<double> packed_qr = original;

    const std::vector<double> tau =
        hdos::detail::QR_decomp(
            packed_qr,
            3,
            2
        );

    std::vector<double> transformed = original;

    hdos::detail::apply_Q_transpose(
        packed_qr,
        tau,
        3,
        2,
        transformed,
        2
    );

    // Q^T A should be:
    //
    // [-5  -2       ]
    // [ 0  -sqrt(5) ]
    // [ 0   0       ]
    const std::vector<double> expected{
        -5.0, 0.0, 0.0,
        -2.0, -std::sqrt(5.0), 0.0
    };

    ASSERT_EQ(transformed.size(), expected.size());

    for (std::size_t k = 0; k < expected.size(); ++k) {
        EXPECT_NEAR(
            transformed[k],
            expected[k],
            1e-12
        );
    }
}

TEST(ApplyQTranspose, AppliesToSingleVector)
{
    const std::vector<double> original_matrix{
        4.0, 3.0, 0.0,
        1.0, 2.0, 2.0
    };

    std::vector<double> packed_qr = original_matrix;

    const std::vector<double> tau =
        hdos::detail::QR_decomp(
            packed_qr,
            3,
            2
        );

    // This is the first column of the original matrix.
    std::vector<double> vector{
        4.0,
        3.0,
        0.0
    };

    hdos::detail::apply_Q_transpose(
        packed_qr,
        tau,
        3,
        2,
        vector,
        1
    );

    EXPECT_NEAR(vector[0], -5.0, 1e-12);
    EXPECT_NEAR(vector[1], 0.0, 1e-12);
    EXPECT_NEAR(vector[2], 0.0, 1e-12);
}

TEST(ApplyQTranspose, LeavesMatrixUnchangedWhenQIsIdentity)
{
    // Already upper triangular with positive diagonal,
    // so every reflector has tau == 0.
    std::vector<double> packed_qr{
        2.0, 0.0, 0.0,
        1.0, 3.0, 0.0,
        4.0, 5.0, 6.0
    };

    const std::vector<double> tau =
        hdos::detail::QR_decomp(
            packed_qr,
            3,
            3
        );

    const std::vector<double> original_B{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };

    std::vector<double> B = original_B;

    hdos::detail::apply_Q_transpose(
        packed_qr,
        tau,
        3,
        3,
        B,
        2
    );

    EXPECT_EQ(B, original_B);
}


TEST(ApplyQTranspose, RejectsInvalidTauDimension)
{
    const std::vector<double> packed_qr(6, 0.0);
    const std::vector<double> tau(1, 0.0);
    std::vector<double> B(3, 0.0);

    // min(3, 2) == 2, but tau contains only one entry.
    EXPECT_THROW(
        hdos::detail::apply_Q_transpose(
            packed_qr,
            tau,
            3,
            2,
            B,
            1
        ),
        std::invalid_argument
    );
}

TEST(ApplyQTranspose, RejectsInvalidTargetDimension)
{
    const std::vector<double> packed_qr(6, 0.0);
    const std::vector<double> tau(2, 0.0);

    // A 3 x 2 target requires six entries.
    std::vector<double> B(5, 0.0);

    EXPECT_THROW(
        hdos::detail::apply_Q_transpose(
            packed_qr,
            tau,
            3,
            2,
            B,
            2
        ),
        std::invalid_argument
    );
}



TEST(ApplyQ, ReconstructsOriginalMatrixFromPaddedR)
{
    // Original column-major matrix:
    //
    // [4  1]
    // [3  2]
    // [0  2]
    const std::vector<double> original{
        4.0, 3.0, 0.0,
        1.0, 2.0, 2.0
    };

    std::vector<double> packed_qr = original;

    const std::vector<double> tau =
        hdos::detail::QR_decomp(
            packed_qr,
            3,
            2
        );

    // Padded R:
    //
    // [-5  -2       ]
    // [ 0  -sqrt(5) ]
    // [ 0   0       ]
    std::vector<double> padded_R{
        -5.0, 0.0, 0.0,
        -2.0, -std::sqrt(5.0), 0.0
    };

    hdos::detail::apply_Q(
        packed_qr,
        tau,
        3,
        2,
        padded_R,
        2
    );

    ASSERT_EQ(padded_R.size(), original.size());

    for (std::size_t k = 0; k < original.size(); ++k) {
        EXPECT_NEAR(
            padded_R[k],
            original[k],
            1e-12
        );
    }
}


TEST(ApplyQ, InvertsApplyQTranspose)
{
    const std::vector<double> matrix{
        4.0, 3.0, 0.0,
        1.0, 2.0, 2.0
    };

    std::vector<double> packed_qr = matrix;

    const std::vector<double> tau =
        hdos::detail::QR_decomp(
            packed_qr,
            3,
            2
        );

    // Arbitrary 3 x 2 column-major matrix.
    const std::vector<double> original_B{
         1.0, 2.0, 3.0,
        -2.0, 4.0, 1.0
    };

    std::vector<double> B = original_B;

    hdos::detail::apply_Q_transpose(
        packed_qr,
        tau,
        3,
        2,
        B,
        2
    );

    hdos::detail::apply_Q(
        packed_qr,
        tau,
        3,
        2,
        B,
        2
    );

    ASSERT_EQ(B.size(), original_B.size());

    for (std::size_t k = 0; k < original_B.size(); ++k) {
        EXPECT_NEAR(
            B[k],
            original_B[k],
            1e-12
        );
    }
}


TEST(ApplyQ, LeavesMatrixUnchangedWhenQIsIdentity)
{
    std::vector<double> packed_qr{
        2.0, 0.0, 0.0,
        1.0, 3.0, 0.0,
        4.0, 5.0, 6.0
    };

    const std::vector<double> tau =
        hdos::detail::QR_decomp(
            packed_qr,
            3,
            3
        );

    const std::vector<double> original_B{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };

    std::vector<double> B = original_B;

    hdos::detail::apply_Q(
        packed_qr,
        tau,
        3,
        3,
        B,
        2
    );

    EXPECT_EQ(B, original_B);
}

TEST(ApplyQ, RejectsInvalidTauDimension)
{
    const std::vector<double> packed_qr(6, 0.0);
    const std::vector<double> tau(1, 0.0);
    std::vector<double> B(3, 0.0);

    EXPECT_THROW(
        hdos::detail::apply_Q(
            packed_qr,
            tau,
            3,
            2,
            B,
            1
        ),
        std::invalid_argument
    );
}

TEST(ApplyQ, RejectsInvalidTargetDimension)
{
    const std::vector<double> packed_qr(6, 0.0);
    const std::vector<double> tau(2, 0.0);
    std::vector<double> B(5, 0.0);

    EXPECT_THROW(
        hdos::detail::apply_Q(
            packed_qr,
            tau,
            3,
            2,
            B,
            2
        ),
        std::invalid_argument
    );
}


TEST(FormQ, TallMatrix)
{
    const std::vector<double> A{
        4.0, 3.0, 0.0,
        1.0, 2.0, 2.0
    };

    expect_valid_QR(A, 3, 2);
}

TEST(FormQ, SquareMatrix)
{
    // [4  1  2]
    // [3  2  0]
    // [0  2  1]
    const std::vector<double> A{
        4.0, 3.0, 0.0,
        1.0, 2.0, 2.0,
        2.0, 0.0, 1.0
    };

    expect_valid_QR(A, 3, 3);
}

TEST(FormQ, WideMatrix)
{
    // [3  1  2]
    // [4  2  0]
    const std::vector<double> A{
        3.0, 4.0,
        1.0, 2.0,
        2.0, 0.0
    };

    expect_valid_QR(A, 2, 3);
}

TEST(FormQ, RankDeficientMatrix)
{
    // [1  0  1]
    // [2  0  0]
    // [3  0  1]
    // [4  0  0]
    //
    // Second column is identically zero.
    const std::vector<double> A{
        1.0, 2.0, 3.0, 4.0,
        0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 1.0, 0.0
    };

    expect_valid_QR(A, 4, 3);
}