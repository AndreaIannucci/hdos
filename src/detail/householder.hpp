#pragma once

#include <span>

namespace hdos::detail{
double householder_reflector(std::span<double> x);
void  apply_householder_left(
    std::span<double> A,
    std::size_t rows,
    std::size_t cols,
    std::size_t start_row,
    std::size_t start_col,
    std::span<const double> packed,
    double tau);

void  apply_householder_right(
    std::span<double> A,
    std::size_t rows,
    std::size_t cols,
    std::size_t start_row,
    std::size_t start_col,
    std::span<const double> packed,
    double tau);
}