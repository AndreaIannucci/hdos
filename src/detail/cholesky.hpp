
#pragma once

#include <span>
#include <vector>

namespace hdos::detail{
std::vector<double> cholesky_decomp(std::span<const double> M);
void rk1_cholesky(std::vector<double>& L, std::span<const double> x,  double c = 1,
     const bool intercept = false);
}