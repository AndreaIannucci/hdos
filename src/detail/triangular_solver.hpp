#pragma once

#include <vector>

namespace hdos::detail{
std::vector<double> lower_triangular_solver(
    const std::vector<double>& L,
    const std::vector<double>& b
);

std::vector<double> upper_triangular_solver(
    const std::vector<double>& L,
    const std::vector<double>& b
);
}