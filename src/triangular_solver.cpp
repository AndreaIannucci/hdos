#include "detail/triangular_solver.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>



namespace hdos::detail{
// Implement a lower triangular solver
// @params
//      L: column-major matrix
//      b: right-hand-side vector
std::vector<double> lower_triangular_solver(
    const std::vector<double>& L,
    const std::vector<double>& b
)
{
    const std::size_t N = b.size();
    std::vector<double> solution(N);

    for (std::size_t k = 0; k < N; ++k) {
        double right_side = b[k];

        for (std::size_t j = 0; j < k; ++j) {
            right_side -= L[k + j * N] * solution[j];
        }

        const double diag = L[k + k * N];

        if (diag == 0.0) {
            throw std::invalid_argument("The system is singular");
        }

        solution[k] = right_side / diag;
    }

    return solution;
}


// Implement an upper triangular solver
// @params
//      L: column-major matrix
//      b: right-hand-side vector
std::vector<double> upper_triangular_solver(
    const std::vector<double>& L,
    const std::vector<double>& b
)
{
    const std::size_t N = b.size();
    std::vector<double> solution(N);

    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t k_rev = N - 1 - k;
        double right_side = b[k_rev];

        for (std::size_t j = 0; j < k; ++j) {
            const std::size_t j_rev = N - 1 - j;

            right_side -=
                L[k_rev + j_rev * N] * solution[j_rev];
        }

        const double diag = L[k_rev + k_rev * N];

        if (diag == 0.0) {
            throw std::invalid_argument("The system is singular");
        }

        solution[k_rev] = right_side / diag;
    }

    return solution;
}
}
