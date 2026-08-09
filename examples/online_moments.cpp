#include <hdos/hdos.hpp>

#include <array>
#include <iostream>

int main()
{
    hdos::RunningMean running_mean(2);
    hdos::RunningVariance running_variance(2);

    const std::array<std::array<double, 2>, 4> observations{{
        {1.0, 10.0},
        {2.0, 20.0},
        {3.0, 30.0},
        {4.0, 40.0}
    }};

    for (const auto& observation : observations) {
        running_mean.update(observation);
        running_variance.update(observation);
    }

    const auto mean =
        running_mean.mean();

    const auto variance =
        running_variance.variance();

    std::cout
        << "Observations: "
        << running_mean.n_observations()
        << '\n'
        << "Mean: ["
        << mean[0] << ", "
        << mean[1] << "]\n"
        << "Sample variance: ["
        << variance[0] << ", "
        << variance[1] << "]\n";
}