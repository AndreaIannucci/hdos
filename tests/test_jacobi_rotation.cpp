#include <gtest/gtest.h>

#include "detail/jacobi_rotation.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

double dot(
    const std::vector<double>& x,
    const std::vector<double>& y)
{
    double value = 0.0;

    for (std::size_t k = 0; k < x.size(); ++k) {
        value += x[k] * y[k];
    }

    return value;
}

}



TEST(JacobiRotation, ReturnsIdentityForOrthogonalColumns)
{
    const auto rotation =
        hdos::detail::jacobi_rotation(
            1.0,
            4.0,
            0.0
        );

    EXPECT_EQ(rotation.cosine, 1.0);
    EXPECT_EQ(rotation.sine, 0.0);
}


TEST(JacobiRotation, ProducesUnitRotation)
{
    const auto rotation =
        hdos::detail::jacobi_rotation(
            14.0,
            21.0,
            8.0
        );

    EXPECT_NEAR(
        rotation.cosine * rotation.cosine +
        rotation.sine * rotation.sine,
        1.0,
        1e-12
    );
}


TEST(JacobiRotation, OrthogonalizesTwoColumns)
{
    std::vector<double> x{
        1.0, 2.0, 3.0
    };

    std::vector<double> y{
        4.0, -1.0, 2.0
    };

    const double alpha = dot(x, x);
    const double beta  = dot(y, y);
    const double gamma = dot(x, y);

    const auto rotation =
        hdos::detail::jacobi_rotation(
            alpha,
            beta,
            gamma
        );

    hdos::detail::apply_jacobi_rotation(
        x,
        y,
        rotation.cosine,
        rotation.sine
    );

    EXPECT_NEAR(
        dot(x, y),
        0.0,
        1e-12
    );
}


TEST(JacobiRotation, HandlesNegativeInnerProduct)
{
    std::vector<double> x{
        1.0, 2.0, 0.0
    };

    std::vector<double> y{
        -2.0, 0.0, 1.0
    };

    const double alpha = dot(x, x);
    const double beta  = dot(y, y);
    const double gamma = dot(x, y);

    ASSERT_LT(gamma, 0.0);

    const auto rotation =
        hdos::detail::jacobi_rotation(
            alpha,
            beta,
            gamma
        );

    hdos::detail::apply_jacobi_rotation(
        x,
        y,
        rotation.cosine,
        rotation.sine
    );

    EXPECT_NEAR(
        dot(x, y),
        0.0,
        1e-12
    );
}


TEST(JacobiRotation, PreservesCombinedColumnNorm)
{
    std::vector<double> x{
        1.0, 2.0, 3.0
    };

    std::vector<double> y{
        4.0, -1.0, 2.0
    };

    const double alpha = dot(x, x);
    const double beta  = dot(y, y);
    const double gamma = dot(x, y);

    const double norm_before =
        alpha + beta;

    const auto rotation =
        hdos::detail::jacobi_rotation(
            alpha,
            beta,
            gamma
        );

    hdos::detail::apply_jacobi_rotation(
        x,
        y,
        rotation.cosine,
        rotation.sine
    );

    const double norm_after =
        dot(x, x) + dot(y, y);

    EXPECT_NEAR(
        norm_after,
        norm_before,
        1e-12
    );
}



TEST(JacobiRotation, RejectsDifferentColumnLengths)
{
    std::vector<double> x{
        1.0, 2.0
    };

    std::vector<double> y{
        1.0, 2.0, 3.0
    };

    EXPECT_THROW(
        hdos::detail::apply_jacobi_rotation(
            x,
            y,
            1.0,
            0.0
        ),
        std::invalid_argument
    );
}