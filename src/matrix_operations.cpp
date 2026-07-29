#include "detail/matrix_operations.hpp"

#include<vector>
#include<cstddef>
#include<cmath>


namespace hdos::detail{

    // Transpose matrix
std::vector<double> transpose(
    const std::vector<double>& M,
    const std::size_t rows,
    const std::size_t cols
)
{
    std::vector<double> M_adj(rows * cols);

    for (std::size_t k = 0; k < rows; ++k) {
        for (std::size_t j = 0; j < cols; ++j) {
            M_adj[k * cols + j] = M[k + j * rows];
        }
    }
    return M_adj;
}
}