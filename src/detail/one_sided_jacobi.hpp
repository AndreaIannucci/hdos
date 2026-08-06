#pragma once

#include <vector>
#include <span>
#include <stdexcept>

namespace hdos::detail{
struct SVDResult {
    std::vector<double> U;
    std::vector<double> singular_values;
    std::vector<double> V;
};
SVDResult jacobi_svd(
    std::span<const double> A,
    std::size_t rows,
    std::size_t cols,
    double tol_const = 10,
    std::size_t max_sweeps = 100);
}