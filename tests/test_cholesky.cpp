#include <gtest/gtest.h>

#include "detail/cholesky.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

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
    EXPECT_NEAR(lower[3], std::sqrt(2.0), 1e-12);
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