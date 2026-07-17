#include "detail/cholesky.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>


// TODO:
// Change double to template
// Maintain "triangular" instead of complete matrix
// Check size of x is compatible with L


namespace hdos::detail {

std::vector<double> cholesky_decomp(const std::vector<double>& M)
{
    const std::size_t N =
        static_cast<std::size_t>(std::sqrt(M.size()));

    std::vector<double> L(M.size());

    for (std::size_t k = 0; k < N; ++k) {
        for (std::size_t j = 0; j <= k; ++j) {
            double sum = 0.0;

            for (std::size_t i = 0; i < j; ++i) {
                sum += L[k + i * N] * L[j + i * N];
            }

            if (k == j) {
                const double addend = M[k + N * k] - sum;

                if (addend < 0.0) {
                    throw std::invalid_argument(
                        "The matrix is not positive definite"
                    );
                }

                L[k + N * j] = std::sqrt(addend);
            } else {
                const double denom = L[j + N * j];

                if (denom == 0.0) {
                    throw std::invalid_argument(
                        "The matrix is not positive definite"
                    );
                }

                L[k + N * j] =
                    (M[k + j * N] - sum) / denom;
            }
        }
    }

    return L;
}


void rk1_cholesky(
    std::vector<double>& L,
    std::vector<double> x
)
{
    // Rank-one update of the triangular Cholesky factor L by x.

    double c = 1.0;
    const std::size_t N = x.size();

    for (std::size_t k = 0; k < N - 1; ++k) {
        const auto start = L.begin() + N * k;
        const auto end = L.begin() + N * (k + 1);

        const std::vector<double> l(start, end);

        const double lk = l[k];
        const double xk = x[k];
        const double dk = std::sqrt(lk * lk + c * xk * xk);

        for (std::size_t j = 0; j < N; ++j) {
            L[k + j * N] =
                (lk / dk) * l[j] + (c * xk / dk) * x[j];
        }

        for (std::size_t j = 0; j < N; ++j) {
            x[j] -= l[j] * (xk / lk);
        }

        c *= (lk / dk) * (lk / dk);
    }
        L[(N - 1) + N * (N - 1)] =
            std::sqrt(
                L[(N - 1) + N * (N - 1)]
                    * L[(N - 1) + N * (N - 1)]
                + c * x[N - 1] * x[N - 1]
            );
    
}

} // namespace hdos::detail

// int main(){
//     vector<double> v = {1,2,3,5};
//     vector<double> x = {1,0};
//     vector<double> L = cholesky(v);
//     mat_print(L);
//     return 0;
// }