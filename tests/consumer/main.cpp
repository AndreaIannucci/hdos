#include <hdos/hdos.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

bool near(double left, double right)
{
    return std::abs(left - right) <= 1e-10;
}

int fail(const char* message)
{
    std::cerr << message << '\n';
    return EXIT_FAILURE;
}

}  // namespace

int main()
{
    hdos::LinearRegressionOptions options;
    options.solver =
        hdos::LinearRegressionSolver::svd;

    hdos::LinearRegression regression(1, options);

    const std::vector<double> regression_X{
        0.0,
        1.0,
        2.0,
        3.0
    };

    const std::vector<double> regression_y{
        1.0,
        3.0,
        5.0,
        7.0
    };

    regression.fit(regression_X, regression_y);

    const auto& coefficients =
        regression.coefficients();

    if (coefficients.size() != 1 ||
        !near(coefficients[0], 2.0) ||
        !near(regression.intercept(), 1.0) ||
        !near(regression.r_squared(), 1.0)) {
        return fail("Linear regression smoke test failed");
    }

    hdos::RunningVariance moments(2);

    const std::array<double, 2> first{
        1.0,
        10.0
    };

    const std::array<double, 2> second{
        2.0,
        20.0
    };

    const std::array<double, 2> third{
        3.0,
        30.0
    };

    moments.update(first);
    moments.update(second);
    moments.update(third);

    const auto mean = moments.mean();
    const auto variance = moments.variance();

    if (!near(mean[0], 2.0) ||
        !near(mean[1], 20.0) ||
        !near(variance[0], 1.0) ||
        !near(variance[1], 100.0)) {
        return fail("Online moments smoke test failed");
    }

    const std::vector<double> pca_X{
        -1.0,
         0.0,
         1.0,
         2.0,
        -2.0,
         0.0,
         2.0,
         4.0
    };

    hdos::PCA pca(2, 1);
    pca.fit(pca_X);

    if (pca.n_observations() != 4 ||
        pca.components().size() != 2 ||
        pca.singular_values().size() != 1 ||
        pca.explained_variance().size() != 1) {
        return fail("PCA smoke test failed");
    }

    const std::array<double, 2> new_observation{
        3.0,
        6.0
    };

    pca.update(new_observation);

    if (pca.n_observations() != 5) {
        return fail("Incremental PCA smoke test failed");
    }

    return EXIT_SUCCESS;
}
