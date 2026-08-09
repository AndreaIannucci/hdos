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

    if (L.size() != N * N) {
        throw std::invalid_argument("Incompatible dimensions");
    }

    std::vector<double> solution = b;

    for (std::size_t j = 0; j < N; ++j) {
        const double diag = L[j + j * N];

        if (diag == 0.0) {
            throw std::invalid_argument("The system is singular");
        }

        solution[j] /= diag;
        const double value = solution[j];

        // Contiguous traversal down column j.
        for (std::size_t i = j + 1; i < N; ++i) {
            solution[i] -= L[i + j * N] * value;
        }
    }

    return solution;
}

// Implement an upper triangular solver
// @params
//      L: column-major matrix
//      b: right-hand-side vector
std::vector<double> upper_triangular_solver(
    const std::vector<double>& U,
    const std::vector<double>& b
)
{
    const std::size_t N = b.size();

    if (U.size() != N * N) {
        throw std::invalid_argument("Incompatible dimensions");
    }

    std::vector<double> solution = b;

    for (std::size_t j = N; j-- > 0;) {
        const double diag = U[j + j * N];

        if (diag == 0.0) {
            throw std::invalid_argument("The system is singular");
        }

        solution[j] /= diag;
        const double value = solution[j];

        // Contiguous traversal up column j.
        for (std::size_t i = 0; i < j; ++i) {
            solution[i] -= U[i + j * N] * value;
        }
    }

    return solution;
}

std::vector<double> lower_transpose_solver(
    const std::vector<double>& L,
    const std::vector<double>& b
)
{
    const std::size_t N = b.size();

    if (L.size() != N * N) {
        throw std::invalid_argument("Incompatible dimensions");
    }

    std::vector<double> solution(N);

    for (std::size_t j = N; j-- > 0;) {
        double right_side = b[j];

        // Contiguous traversal down column j of L.
        for (std::size_t i = j + 1; i < N; ++i) {
            right_side -= L[i + j * N] * solution[i];
        }

        const double diag = L[j + j * N];

        if (diag == 0.0) {
            throw std::invalid_argument("The system is singular");
        }

        solution[j] = right_side / diag;
    }

    return solution;
}


}
