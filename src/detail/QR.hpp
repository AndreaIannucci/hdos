#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace hdos::detail {

std::vector<double> QR_decomp(
    std::span<double> A,
    std::size_t rows,
    std::size_t cols);

std::vector<double> extract_R(std::span<const double> packed_qr, std::size_t rows, std::size_t cols);

void apply_Q_transpose(
    std::span<const double> packed_qr,
    std::span<const double> tau,
    std::size_t qr_rows,
    std::size_t qr_cols,
    std::span<double> B,
    std::size_t B_cols);

    void apply_Q(
    std::span<const double> packed_qr,
    std::span<const double> tau,
    std::size_t qr_rows,
    std::size_t qr_cols,
    std::span<double> B,
    std::size_t B_cols);

    std::vector<double> form_Q(
    std::span<const double> packed_qr,
    std::span<const double> tau,
    std::size_t rows,
    std::size_t cols);
}