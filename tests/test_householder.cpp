#include <gtest/gtest.h>

#include "detail/householder.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

TEST(HouseholderReflector, ConstructsReflectorWithPositiveLeadingEntry)
{
    // x = [4, 3]^T.
    //
    // The packed result should represent:
    //
    // beta = -5
    // v    = [1, 1/3]^T
    // tau  = 9/5
    std::vector<double> x{
        4.0, 3.0
    };

    const double tau =
        hdos::detail::householder_reflector(x);

    ASSERT_EQ(x.size(), 2U);
    EXPECT_NEAR(x[0], -5.0, 1e-12);
    EXPECT_NEAR(x[1], 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(tau, 9.0 / 5.0, 1e-12);
}

TEST(HouseholderReflector, ConstructsReflectorWithNegativeLeadingEntry)
{
    // x = [-4, 3]^T.
    //
    // The sign choice gives:
    //
    // beta = 5
    // v    = [1, -1/3]^T
    // tau  = 9/5
    std::vector<double> x{
        -4.0, 3.0
    };

    const double tau =
        hdos::detail::householder_reflector(x);

    ASSERT_EQ(x.size(), 2U);
    EXPECT_NEAR(x[0], 5.0, 1e-12);
    EXPECT_NEAR(x[1], -1.0 / 3.0, 1e-12);
    EXPECT_NEAR(tau, 9.0 / 5.0, 1e-12);
}

TEST(HouseholderReflector, HandlesZeroLeadingEntry)
{
    // x = [0, 3, 4]^T.
    //
    // beta = -5
    // v    = [1, 3/5, 4/5]^T
    // tau  = 1
    std::vector<double> x{
        0.0, 3.0, 4.0
    };

    const double tau =
        hdos::detail::householder_reflector(x);

    ASSERT_EQ(x.size(), 3U);
    EXPECT_NEAR(x[0], -5.0, 1e-12);
    EXPECT_NEAR(x[1], 3.0 / 5.0, 1e-12);
    EXPECT_NEAR(x[2], 4.0 / 5.0, 1e-12);
    EXPECT_NEAR(tau, 1.0, 1e-12);
}

TEST(HouseholderReflector, ReturnsIdentityForZeroTail)
{
    // No entries need to be eliminated.
    std::vector<double> x{
        5.0, 0.0, 0.0
    };

    const double tau =
        hdos::detail::householder_reflector(x);

    EXPECT_EQ(tau, 0.0);
    EXPECT_EQ(x[0], 5.0);
    EXPECT_EQ(x[1], 0.0);
    EXPECT_EQ(x[2], 0.0);
}

TEST(HouseholderReflector, ReturnsIdentityForZeroVector)
{
    std::vector<double> x{
        0.0, 0.0, 0.0
    };

    const double tau =
        hdos::detail::householder_reflector(x);

    EXPECT_EQ(tau, 0.0);
    EXPECT_EQ(x[0], 0.0);
    EXPECT_EQ(x[1], 0.0);
    EXPECT_EQ(x[2], 0.0);
}

TEST(HouseholderReflector, ReturnsIdentityForOneDimensionalVector)
{
    std::vector<double> x{
        -7.0
    };

    const double tau =
        hdos::detail::householder_reflector(x);

    ASSERT_EQ(x.size(), 1U);
    EXPECT_EQ(tau, 0.0);
    EXPECT_EQ(x[0], -7.0);
}

TEST(HouseholderReflector, RejectsEmptyInput)
{
    std::vector<double> x;

    EXPECT_THROW(
        hdos::detail::householder_reflector(x),
        std::invalid_argument
    );
}

TEST(HouseholderReflector, RejectsNaNInput)
{
    std::vector<double> x{
        1.0,
        std::numeric_limits<double>::quiet_NaN()
    };

    EXPECT_THROW(
        hdos::detail::householder_reflector(x),
        std::invalid_argument
    );
}

TEST(HouseholderReflector, RejectsInfiniteInput)
{
    std::vector<double> x{
        1.0,
        std::numeric_limits<double>::infinity()
    };

    EXPECT_THROW(
        hdos::detail::householder_reflector(x),
        std::invalid_argument
    );
}

TEST(HouseholderReflector, HandlesVeryLargeValues)
{
    std::vector<double> x{
        1.0e300,
        -2.0e300,
        3.0e300
    };

    const double tau =
        hdos::detail::householder_reflector(x);

    EXPECT_TRUE(std::isfinite(x[0]));
    EXPECT_TRUE(std::isfinite(x[1]));
    EXPECT_TRUE(std::isfinite(x[2]));
    EXPECT_TRUE(std::isfinite(tau));

    EXPECT_GE(tau, 1.0);
    EXPECT_LE(tau, 2.0);
}

TEST(HouseholderReflector, HandlesVerySmallValues)
{
    std::vector<double> x{
        1.0e-300,
        -2.0e-300,
        3.0e-300
    };

    const double tau =
        hdos::detail::householder_reflector(x);

    EXPECT_TRUE(std::isfinite(x[0]));
    EXPECT_TRUE(std::isfinite(x[1]));
    EXPECT_TRUE(std::isfinite(x[2]));
    EXPECT_TRUE(std::isfinite(tau));

    EXPECT_NE(x[0], 0.0);
    EXPECT_GE(tau, 1.0);
    EXPECT_LE(tau, 2.0);
}