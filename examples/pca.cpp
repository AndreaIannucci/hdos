#include <hdos/hdos.hpp>

#include <iostream>
#include <vector>

int main()
{
    /*
     * Six observations and two highly correlated features,
     * stored column-major.
     */
    const std::vector<double> X{
        -3.0, -2.0, -1.0, 1.0, 2.0, 3.0,
        -3.1, -1.9, -1.1, 0.9, 2.1, 3.0
    };

    hdos::PCA model(2, 1);
    model.fit(X);

    const auto mean =
        model.mean();

    const auto components =
        model.components();

    const auto singular_values =
        model.singular_values();

    const auto explained_variance =
        model.explained_variance();

    std::cout
        << "Observations: "
        << model.n_observations()
        << '\n'
        << "Mean: ["
        << mean[0] << ", "
        << mean[1] << "]\n"
        << "First principal component: ["
        << components[0] << ", "
        << components[1] << "]\n"
        << "Singular value: "
        << singular_values[0]
        << '\n'
        << "Explained variance: "
        << explained_variance[0]
        << '\n';

    const std::vector<double> new_observation{
        4.0,
        4.1
    };

    model.update(new_observation);

    std::cout
        << "Observations after update: "
        << model.n_observations()
        << '\n';
}