#pragma once

#include <vector>
namespace hdos::detail{

std::vector<double> solve_pos_definite_cholesky(
    const std::vector<double>& L,
    const std::vector<double>& b
);

std::vector<double> invert_pos_def_cholesky(
    const std::vector<double>& L
);

std::vector<double> invert_pos_def(
    const std::vector<double>& M
);
}

