#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "detail/one_sided_jacobi.hpp"

namespace {

void expect_reconstruction(
    const std::vector<double>& A,
    const hdos::detail::SVDResult& svd,
    std::size_t rows,
    std::size_t cols,
    double tol = 1e-10)
{
    const std::size_t rank = std::min(rows, cols);

    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {

            double reconstructed = 0.0;

            for (std::size_t k = 0; k < rank; ++k) {
                reconstructed +=
                    svd.U[k * rows + i]
                    * svd.singular_values[k]
                    * svd.V[k * cols + j];
            }

            EXPECT_NEAR(
                reconstructed,
                A[j * rows + i],
                tol);
        }
    }
}


void expect_orthonormal_columns(
    const std::vector<double>& A,
    std::size_t rows,
    std::size_t cols,
    double tol = 1e-10)
{
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t k = 0; k < cols; ++k) {

            double inner_product = 0.0;

            for (std::size_t i = 0; i < rows; ++i) {
                inner_product +=
                    A[j * rows + i] *
                    A[k * rows + i];
            }

            const double expected = (j == k) ? 1.0 : 0.0;

            EXPECT_NEAR(inner_product, expected, tol);
        }
    }
}

}

TEST(JacobiSVDTest, WideRankDeficientMatrix)
{
    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 4;

    // [1 2 3 4]
    // [2 4 6 8]
    std::vector<double> A{
        1.0, 2.0,
        2.0, 4.0,
        3.0, 6.0,
        4.0, 8.0
    };

    auto svd = hdos::detail::SVD(A, rows, cols);

    expect_reconstruction(A, svd, rows, cols);

    EXPECT_NEAR(
        svd.singular_values[0],
        std::sqrt(150.0),
        1e-10);

    EXPECT_NEAR(
        svd.singular_values[1],
        0.0,
        1e-10);
}

