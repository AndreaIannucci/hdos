#include <vector>
#include <span>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <utility>

#include "detail/one_sided_jacobi.hpp"
#include "detail/stable_norm.hpp"
#include "detail/jacobi_rotation.hpp"

namespace hdos::detail{
SVDResult jacobi_svd(
    std::span<const double> A,
    std::size_t rows,
    std::size_t cols,
    double tol_const,
    std::size_t max_sweeps){
    
    if (rows == 0 || cols == 0){
        throw std::invalid_argument("The matrix cannot have 0 rows or 0 columns");
        }

    if (A.size() != rows * cols){
        throw std::invalid_argument("Invalid matrix dimensions");
    }

    if (rows < cols){
        throw std::invalid_argument("Matrix cannot have rows < cols");
        }

    for (std::size_t k =0; k < A.size(); ++k){
        if (!std::isfinite(A[k])){
            throw std::invalid_argument("No entry can be nan or infty");
        }
    }

    if (tol_const <= 0.0){
        throw std::invalid_argument("Tolerance constant has to be strictly positive");
    }

    std::vector<double> B(A.begin(), A.end());
    std::vector<double> V(cols*cols, 0.0);

    for (std::size_t k =0; k<cols; ++k){
        V[k + k*cols] = 1;
    }

    double epsilon = std::numeric_limits<double>::epsilon();
    double tolerance = tol_const * rows * epsilon;
    bool converged = false;

    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep){
        bool rotated = false;
        for (std::size_t p = 0; p + 1 < cols; ++p){
            for (std::size_t q = p+1; q < cols; ++q){
                std::span<double> x(B.data() + p * rows, rows);
                std::span<double> y(B.data() + q * rows, rows);
                
                double norm_x = detail::stable_norm(x);
                double norm_y = detail::stable_norm(y);

                if (norm_x == 0 or norm_y == 0){
                    continue;
                }
                
                double scale = std::max(norm_x, norm_y);

                double scaled_norm_x = norm_x / scale;
                double scaled_norm_y = norm_y / scale;

                double alpha = scaled_norm_x * scaled_norm_x;

                double beta =
                    scaled_norm_y * scaled_norm_y;

                double gamma = 0;

                for (std::size_t i = 0; i < rows; ++i){

                    gamma += (x[i] / scale) * (y[i] / scale);
                }

                double threshold = tolerance * scaled_norm_x * scaled_norm_y;

                if (std::abs(gamma) <= threshold){
                    continue;
                }

                JacobiRotation rotation = jacobi_rotation( alpha, beta, gamma);
                apply_jacobi_rotation(x, y, rotation.cosine, rotation.sine);

                std::span<double> v_p(V.data() + p * cols, cols); 
                std::span<double> v_q(V.data() + q * cols, cols);


                apply_jacobi_rotation(v_p, v_q, rotation.cosine, rotation.sine);
                rotated = true;
            }
        }
            if (rotated == false){
                converged = true;
                break;
             
        }
    }
    if (converged == false){
        throw std::runtime_error("Jacobi SVD failed to converge" );
    }

    std::vector<double> singular_values(cols);

    for (std::size_t j = 0; j < cols; ++j) {
        std::span<const double> column(B.data() + j * rows, rows);

        singular_values[j] = detail::stable_norm(column);
    }
    
    for (std::size_t j = 0; j < cols; ++j) {

    // Find the largest remaining singular value.
    std::size_t largest = j;

    for (std::size_t k = j + 1; k < cols; ++k) {
        if (singular_values[k] > singular_values[largest])
        {
            largest = k;
        }
    }

    if (largest == j) {
        continue;
    }

    // Swap singular values.
    std::swap(singular_values[j], singular_values[largest]);

    // Swap corresponding columns of B.
    // B has dimensions rows x cols.
    for (std::size_t i = 0; i < rows; ++i) {
        std::swap(B[i + j * rows], B[i + largest * rows]);
    }

    // Swap corresponding columns of V.
    // V has dimensions cols x cols.
    for (std::size_t i = 0; i < cols; ++i) {
        std::swap( V[i + j * cols], V[i + largest * cols]);
        }
    }
    for (std::size_t j = 0; j < cols; ++j){

        double sigma = singular_values[j];

        if (sigma == 0){
            continue;
        }

        for (std::size_t i = 0; i < rows; ++i){
            B[i + j * rows] /= sigma;
        }
    }
    std::vector<double> U = std::move(B);

    return SVDResult{std::move(U), std::move(singular_values), std::move(V)};
}
}