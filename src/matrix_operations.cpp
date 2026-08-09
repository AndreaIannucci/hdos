#include "detail/matrix_operations.hpp"

#include<vector>
#include<cstddef>
#include<cmath>
#include<stdexcept>


namespace hdos::detail{

// Transpose matrix
std::vector<double> transpose(
    std::span<const double> M,
    std::size_t rows,
    std::size_t cols)
{
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument(
            "The matrix cannot have 0 rows or 0 columns");
    }

    if (M.size() != rows * cols) {
        throw std::invalid_argument(
            "Invalid matrix dimensions");
    }

    std::vector<double> M_T(rows * cols);

    for (std::size_t col = 0; col < cols; ++col) {
        for (std::size_t row = 0; row < rows; ++row) {

            M_T[col + row * cols] =
                M[row + col * rows];
        }
    }

    return M_T;
}
}