#include <gtest/gtest.h>

#include "detail/triangular_solver.hpp"

#include <stdexcept>
#include <vector>

TEST(LowerTriangularSolver, SolvesNonDiagonalSystem)
{
    // Column-major representation of:
    //
    // [2   0  0]
    // [1   3  0]
    // [4  -2  1]
    const std::vector<double> lower{
         2.0, 1.0,  4.0,
         0.0, 3.0, -2.0,
         0.0, 0.0,  1.0
    };

    // Generated using x = [1, 2, -1].
    const std::vector<double> right_side{
        2.0, 7.0, -1.0
    };

    const std::vector<double> solution =
        hdos::detail::lower_triangular_solver(
            lower,
            right_side
        );

    ASSERT_EQ(solution.size(), 3U);
    EXPECT_NEAR(solution[0],  1.0, 1e-12);
    EXPECT_NEAR(solution[1],  2.0, 1e-12);
    EXPECT_NEAR(solution[2], -1.0, 1e-12);
}

TEST(LowerTriangularSolver, RejectsSingularSystem)
{
    // [1  0]
    // [2  0]
    const std::vector<double> lower{
        1.0, 2.0,
        0.0, 0.0
    };

    const std::vector<double> right_side{
        1.0, 2.0
    };

    EXPECT_THROW(
        hdos::detail::lower_triangular_solver(
            lower,
            right_side
        ),
        std::invalid_argument
    );
}

TEST(LowerTransposeSolver, SolvesTransposedSystem)
{
    // Column-major representation of:
    //
    // [2   0  0]
    // [1   3  0]
    // [4  -2  1]
    const std::vector<double> lower{
         2.0, 1.0,  4.0,
         0.0, 3.0, -2.0,
         0.0, 0.0,  1.0
    };

    // L^T * [1, 2, -1]^T = [0, 8, -1]^T.
    const std::vector<double> right_side{
        0.0, 8.0, -1.0
    };

    const std::vector<double> solution =
        hdos::detail::lower_transpose_solver(
            lower,
            right_side
        );

    ASSERT_EQ(solution.size(), 3U);
    EXPECT_NEAR(solution[0],  1.0, 1e-12);
    EXPECT_NEAR(solution[1],  2.0, 1e-12);
    EXPECT_NEAR(solution[2], -1.0, 1e-12);
}