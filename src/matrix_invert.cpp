#include "detail/matrix_invert.hpp"
#include "detail/cholesky.hpp"
#include "detail/triangular_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>



namespace hdos::detail{
// Solve positive definite system from Cholesky decomposition
std::vector<double> solve_pos_definite_cholesky(
    const std::vector<double>& L,
    const std::vector<double>& b
)
{
    // (L L*)x = b:
    // first solve Ly = b, then solve L*x = y.

    const std::size_t N =
        static_cast<std::size_t>(std::sqrt(L.size()));

    const std::vector<double> y =
        lower_triangular_solver(L, b);

    const std::vector<double> x =
        lower_transpose_solver(L, y);

    return x;
}


// Invert from an existing Cholesky decomposition
std::vector<double> invert_pos_def_cholesky(
    const std::vector<double>& L
)
{
    const std::size_t N =
        static_cast<std::size_t>(std::sqrt(L.size()));

    std::vector<double> L_inv(L.size());
    std::vector<double> e_k(N, 0.0);

    e_k[0] = 1.0;

    for (std::size_t k = 0; k < N; ++k) {
        const std::vector<double> L_inv_k_col =
            solve_pos_definite_cholesky(L, e_k);

        std::copy(
            L_inv_k_col.begin(),
            L_inv_k_col.end(),
            L_inv.begin() + k * N
        );

        if (k < N - 1) {
            e_k[k] = 0.0;
            e_k[k + 1] = 1.0;
        }
    }

    return L_inv;
}


// Invert a general positive definite matrix
std::vector<double> invert_pos_def(
    const std::vector<double>& M
)
{
    const std::vector<double> L = cholesky_decomp(M);
    return invert_pos_def_cholesky(L);
}
}
