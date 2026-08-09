#include "detail/cholesky.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <span>

// TODO:
// Change double to template
// Maintain "triangular" instead of complete matrix
// Check size of x is compatible with L


namespace hdos::detail {

void cholesky_decomp_in_place(
    std::span<double> M)
{
    const std::size_t N =
        static_cast<std::size_t>(
            std::sqrt(M.size())
        );

    if (N * N != M.size()) {
        throw std::invalid_argument(
            "The matrix must be square"
        );
    }

    for (std::size_t j = 0; j < N; ++j) {
        // Subtract contributions from previously computed columns.
        for (std::size_t k = 0; k < j; ++k) {
            const double factor = M[j + k * N];

            for (std::size_t i = j; i < N; ++i) {
                M[i + j * N] -=
                    M[i + k * N] * factor;
            }
        }

        const double diagonal = M[j + j * N];

        if (!(diagonal > 0.0)) {
            throw std::invalid_argument(
                "The matrix is not positive definite"
            );
        }

        M[j + j * N] = std::sqrt(diagonal);

        // Scale the part below the diagonal.
        for (std::size_t i = j + 1; i < N; ++i) {
            M[i + j * N] /= M[j + j * N];
        }
    }
}


std::vector<double> cholesky_decomp(
    std::span<const double> M)
{
    std::vector<double> L(
        M.begin(),
        M.end()
    );

    cholesky_decomp_in_place(L);

    // Optional: clear the upper triangle so the returned
    // vector visibly contains only the lower factor.
    const std::size_t N =
        static_cast<std::size_t>(
            std::sqrt(L.size())
        );

    for (std::size_t column = 1; column < N; ++column) {
        for (std::size_t row = 0; row < column; ++row) {
            L[row + column * N] = 0.0;
        }
    }

    return L;
}



// std::vector<double> cholesky_decomp(
//     std::span<const double> M)
// {
//     const std::size_t N =
//         static_cast<std::size_t>(std::sqrt(M.size()));

//     if (N * N != M.size()) {
//         throw std::invalid_argument("The matrix must be square");
//     }

//     // M remains unchanged. The upper triangle of L starts at zero.
//     std::vector<double> L(M.size(), 0.0);

//     for (std::size_t j = 0; j < N; ++j) {
//         // Copy only the required lower part of column j.
//         for (std::size_t i = j; i < N; ++i) {
//             L[i + j * N] = M[i + j * N];
//         }

//         // Subtract contributions from previously computed columns.
//         for (std::size_t k = 0; k < j; ++k) {
//             const double factor = L[j + k * N];

//             for (std::size_t i = j; i < N; ++i) {
//                 L[i + j * N] -=
//                     L[i + k * N] * factor;
//             }
//         }

//         const double diagonal = L[j + j * N];

//         if (!(diagonal > 0.0)) {
//             throw std::invalid_argument(
//                 "The matrix is not positive definite"
//             );
//         }

//         L[j + j * N] = std::sqrt(diagonal);

//         // Scale the part below the diagonal.
//         for (std::size_t i = j + 1; i < N; ++i) {
//             L[i + j * N] /= L[j + j * N];
//         }
//     }

//     return L;
// }


// void rk1_cholesky(
//     std::vector<double>& L,
//     std::span<const double> x,
//     
// )
// {
//     // Rank-one update of the triangular Cholesky factor L by x.

//     double c = 1.0;
//     
//     const std::size_t N = x.size();

//     for (std::size_t k = 0; k < N - 1; ++k) {
//         const auto start = L.begin() + N * k;
//         const auto end = L.begin() + N * (k + 1);

//         const std::vector<double> l(start, end);

//         const double lk = l[k];
//         const double xk = x[k];
//         const double dk = std::sqrt(lk * lk + c * xk * xk);

//         for (std::size_t j = 0; j < N; ++j) {
//             L[k + j * N] =
//                 (lk / dk) * l[j] + (c * xk / dk) * x[j];
//         }

//         for (std::size_t j = 0; j < N; ++j) {
//             x[j] -= l[j] * (xk / lk);
//         }

//         c *= (lk / dk) * (lk / dk);
//     }
//         L[(N - 1) + N * (N - 1)] =
//             std::sqrt(
//                 L[(N - 1) + N * (N - 1)]
//                     * L[(N - 1) + N * (N - 1)]
//                 + c * x[N - 1] * x[N - 1]
//             );
//         
// }

void rk1_cholesky(
    std::vector<double>& L,
    std::span<const double> x,
    double c,
    const bool intercept)
{
    const std::size_t p = x.size();
    const std::size_t N =
        p + static_cast<std::size_t>(intercept);

    if (N == 0) {
        throw std::invalid_argument(
            "The updated vector cannot be empty");
    }

    const std::size_t last = N - 1;

    if (L.size() != N * N) {
        throw std::invalid_argument(
            "L has incompatible dimensions");
    }

    if (c < 0.0) {
        throw std::invalid_argument(
            "This routine implements an update, not a downdate");
    }

    if (c == 0.0) {
        return;
    }

    double working_last =
        intercept ? 1.0 : x[last];

    for (std::size_t k = 0; k < last; ++k) {
        const double lkk = L[k + k * N];

        if (lkk <= 0.0) {
            throw std::domain_error(
                "L must have a strictly positive diagonal");
        }

        const double xk =
            (k == 0) ? x[k] : L[k + last * N];

        const double dk =
            std::sqrt(lkk * lkk + c * xk * xk);

        const double col_scale = lkk / dk;
        const double update_scale = c * xk / dk;
        const double elimination_scale = xk / lkk;

        for (std::size_t i = k; i < N; ++i) {
            double xi;

            if (i == last) {
                xi = working_last;
            } else if (k == 0) {
                xi = x[i];
            } else {
                xi = L[i + last * N];
            }

            const double lik = L[i + k * N];

            L[i + k * N] =
                col_scale * lik + update_scale * xi;

            const double transformed_xi =
                xi - elimination_scale * lik;

            if (i == last) {
                working_last = transformed_xi;
            } else {
                L[i + last * N] = transformed_xi;
            }
        }

        c *= col_scale * col_scale;
    }

    L[last + last * N] = std::sqrt(
        L[last + last * N] * L[last + last * N]
        + c * working_last * working_last
    );

    for (std::size_t i = 0; i < last; ++i) {
        L[i + last * N] = 0.0;
    }
}




} // namespace hdos::detail

// int main(){
//     vector<double> v = {1,2,3,5};
//     vector<double> x = {1,0};
//     vector<double> L = cholesky(v);
//     mat_print(L);
//     return 0;
// }