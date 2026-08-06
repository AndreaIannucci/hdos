#include <gtest/gtest.h>

#include "detail/one_sided_jacobi.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void expect_orthonormal_columns(
    const std::vector<double>& A,
    std::size_t rows,
    std::size_t cols,
    double tolerance = 1e-10)
{
    ASSERT_EQ(A.size(), rows * cols);

    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t k = 0; k < cols; ++k) {

            double dot = 0.0;

            for (std::size_t i = 0; i < rows; ++i) {
                dot +=
                    A[i + j * rows] *
                    A[i + k * rows];
            }

            EXPECT_NEAR(
                dot,
                (j == k) ? 1.0 : 0.0,
                tolerance
            );
        }
    }
}


void expect_svd_reconstruction(
    const std::vector<double>& original,
    const hdos::detail::SVDResult& svd,
    std::size_t rows,
    std::size_t cols,
    double tolerance = 1e-10)
{
    ASSERT_EQ(svd.U.size(), rows * cols);
    ASSERT_EQ(svd.singular_values.size(), cols);
    ASSERT_EQ(svd.V.size(), cols * cols);

    // A_ij = sum_k U_ik sigma_k V_jk
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {

            double value = 0.0;

            for (std::size_t k = 0; k < cols; ++k) {
                value +=
                    svd.U[i + k * rows] *
                    svd.singular_values[k] *
                    svd.V[j + k * cols];
            }

            EXPECT_NEAR(
                value,
                original[i + j * rows],
                tolerance
            );
        }
    }
}


void expect_sorted_singular_values(
    const std::vector<double>& singular_values)
{
    for (double sigma : singular_values) {
        EXPECT_GE(sigma, 0.0);
    }

    for (std::size_t k = 1;
         k < singular_values.size();
         ++k)
    {
        EXPECT_GE(
            singular_values[k - 1],
            singular_values[k]
        );
    }
}

}


TEST(JacobiSVD, DiagonalMatrix)
{
    // [3  0]
    // [0  4]
    // [0  0]
    const std::vector<double> A{
        3.0, 0.0, 0.0,
        0.0, 4.0, 0.0
    };

    const auto svd =
        hdos::detail::jacobi_svd(A, 3, 2);

    ASSERT_EQ(svd.singular_values.size(), 2U);

    EXPECT_NEAR(
        svd.singular_values[0],
        4.0,
        1e-12
    );

    EXPECT_NEAR(
        svd.singular_values[1],
        3.0,
        1e-12
    );

    expect_svd_reconstruction(A, svd, 3, 2);
    expect_orthonormal_columns(svd.U, 3, 2);
    expect_orthonormal_columns(svd.V, 2, 2);
    expect_sorted_singular_values(
        svd.singular_values);
}


TEST(JacobiSVD, NontrivialTallMatrix)
{
    // [4  1]
    // [3  2]
    // [0  2]
    const std::vector<double> A{
        4.0, 3.0, 0.0,
        1.0, 2.0, 2.0
    };

    const auto svd =
        hdos::detail::jacobi_svd(A, 3, 2);

    expect_svd_reconstruction(A, svd, 3, 2);
    expect_orthonormal_columns(svd.U, 3, 2);
    expect_orthonormal_columns(svd.V, 2, 2);
    expect_sorted_singular_values(
        svd.singular_values);
}

TEST(JacobiSVD, NontrivialSquareMatrix)
{
    // [4  1  2]
    // [3  2  0]
    // [0  2  1]
    const std::vector<double> A{
        4.0, 3.0, 0.0,
        1.0, 2.0, 2.0,
        2.0, 0.0, 1.0
    };

    const auto svd =
        hdos::detail::jacobi_svd(A, 3, 3);

    expect_svd_reconstruction(A, svd, 3, 3);
    expect_orthonormal_columns(svd.U, 3, 3);
    expect_orthonormal_columns(svd.V, 3, 3);
    expect_sorted_singular_values(
        svd.singular_values);
}


TEST(JacobiSVD, RankDeficientMatrix)
{
    // Two identical columns.
    //
    // [1  1]
    // [2  2]
    // [3  3]
    const std::vector<double> A{
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0
    };

    const auto svd =
        hdos::detail::jacobi_svd(A, 3, 2);

    EXPECT_NEAR(
        svd.singular_values[0],
        std::sqrt(28.0),
        1e-10
    );

    EXPECT_NEAR(
        svd.singular_values[1],
        0.0,
        1e-10
    );

    expect_svd_reconstruction(A, svd, 3, 2);
    expect_orthonormal_columns(svd.V, 2, 2);

    // Don't test full U^T U yet:
    // zero-singular-value U column is currently
    // deliberately left as zero.
}

TEST(JacobiSVD, SingleColumn)
{
    const std::vector<double> A{
        3.0, 4.0, 0.0
    };

    const auto svd =
        hdos::detail::jacobi_svd(A, 3, 1);

    ASSERT_EQ(svd.singular_values.size(), 1U);

    EXPECT_NEAR(
        svd.singular_values[0],
        5.0,
        1e-12
    );

    expect_svd_reconstruction(A, svd, 3, 1);
    expect_orthonormal_columns(svd.U, 3, 1);
    expect_orthonormal_columns(svd.V, 1, 1);
}


TEST(JacobiSVD, RejectsWideMatrix)
{
    const std::vector<double> A(6, 1.0);

    EXPECT_THROW(
        hdos::detail::jacobi_svd(A, 2, 3),
        std::invalid_argument
    );
}

TEST(JacobiSVD, RejectsInvalidDimensions)
{
    const std::vector<double> A(5, 1.0);

    EXPECT_THROW(
        hdos::detail::jacobi_svd(A, 3, 2),
        std::invalid_argument
    );
}

TEST(JacobiSVD, RejectsNonFiniteInput)
{
    const std::vector<double> A{
        1.0, 0.0,
        std::numeric_limits<double>::infinity(),
        1.0
    };

    EXPECT_THROW(
        hdos::detail::jacobi_svd(A, 2, 2),
        std::invalid_argument
    );
}



