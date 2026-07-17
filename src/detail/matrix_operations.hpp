#pragma once

#include <vector>

namespace hdos::detail{
    std::vector<double> transpose(
    const std::vector<double>& M,
    const std::size_t rows,
    const std::size_t cols
);
}