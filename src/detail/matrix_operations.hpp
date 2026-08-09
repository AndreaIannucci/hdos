#pragma once

#include <vector>
#include <span>

namespace hdos::detail{
    std::vector<double> transpose(
    const std::span<const double> M,
    const std::size_t rows,
    const std::size_t cols
);
}