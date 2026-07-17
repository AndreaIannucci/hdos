
#pragma once

#include <vector>

namespace hdos::detail{
std::vector<double> cholesky_decomp(const std::vector<double>& M);
void rk1_cholesky(std::vector<double>& L, std::vector<double> x);
}