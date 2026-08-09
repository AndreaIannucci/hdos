#include <vector>
#include <span>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <utility>

#include "detail/matrix_operations.hpp"
#include "detail/one_sided_jacobi.hpp"
#include "detail/stable_norm.hpp"
#include "detail/jacobi_rotation.hpp"
#include "detail/QR.hpp"

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

    for (double value : A) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "The matrix must contain only finite entries");
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

                const double scale = std::max(norm_x, norm_y);

                // Both columns are exactly zero.
                if (scale == 0.0) {
                    continue;
                }

                // If one column is negligible relative to the other,
                // treat it as a numerical null direction.
                if (std::min(norm_x, norm_y) <= tolerance * scale) {
                    continue;
                }
                

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
    const double rank_tol =
    tolerance * singular_values[0];

    std::vector<double> candidate(rows);
    std::vector<double> best_candidate(rows);

    for (std::size_t j = 0; j < cols; ++j) {

        const double sigma = singular_values[j];

        // Normal singular direction.
        if (sigma > rank_tol) {
            for (std::size_t i = 0; i < rows; ++i) {
                B[i + j * rows] /= sigma;
            }

            continue;
        }

        // Numerically zero singular value.
        singular_values[j] = 0.0;

        double best_norm = 0.0;

        // Find a standard basis vector whose projection onto
        // the orthogonal complement of the previous columns
        // has sufficiently large norm.
        for (std::size_t basis = 0; basis < rows; ++basis) {

            std::fill(candidate.begin(), candidate.end(), 0.0);
            candidate[basis] = 1.0;

            // Two passes of modified Gram-Schmidt for stability.
            for (std::size_t pass = 0; pass < 2; ++pass) {

                for (std::size_t k = 0; k < j; ++k) {

                    double dot = 0.0;

                    for (std::size_t i = 0; i < rows; ++i) {
                        dot +=
                            B[i + k * rows] *
                            candidate[i];
                    }

                    for (std::size_t i = 0; i < rows; ++i) {
                        candidate[i] -=
                            dot * B[i + k * rows];
                    }
                }
            }

            const double candidate_norm =
                detail::stable_norm(
                    std::span<const double>(
                        candidate.data(),
                        candidate.size()));

            if (candidate_norm > best_norm) {
                best_norm = candidate_norm;
                best_candidate = candidate;
            }
        }

        if (best_norm <=
            10.0 * std::numeric_limits<double>::epsilon()) {
            throw std::runtime_error(
                "Failed to construct orthonormal basis in SVD");
        }

        // Store the new unit vector as column j of U.
        for (std::size_t i = 0; i < rows; ++i) {
            B[i + j * rows] =
                best_candidate[i] / best_norm;
        }
    }
    std::vector<double> U = std::move(B);

    return SVDResult{std::move(U), std::move(singular_values), std::move(V)};
}

SVDResult SVD(std::span<const double> A,
              std::size_t rows,
              std::size_t cols,
              double tol_const,
              std::size_t max_sweeps){
            
            if (rows >= cols){
                return jacobi_svd(
                    A,
                    rows,
                    cols,
                    tol_const,
                    max_sweeps);}
                
            
            for (double value : A) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument(
                        "The matrix must contain only finite entries");
                }
            }

            std::vector<double> At = detail::transpose(A, rows, cols);
            std::vector<double> tau =  QR_decomp(At, cols, rows);
            std::vector<double> R(rows*rows, 0.0);

            for (std::size_t j = 0; j < rows; ++j){
                for( std::size_t i = 0; i <= j; ++i){
                    R[i + j*rows] = At[i + j*cols];
                }
            }

            SVDResult svd_R = jacobi_svd(R, rows, rows, tol_const, max_sweeps);
            std::vector<double> U_A = svd_R.V;
            std::vector<double> singular_values = svd_R.singular_values;

            std::vector<double> W(rows*cols, 0.0);
            for (std::size_t j = 0; j < rows; ++j){
                for (std::size_t i = 0; i < rows; ++i){
                    W[i + j*cols] = svd_R.U[i + j *rows];
                }
            }

            for (std::size_t k = rows; k-- > 0;) {

                // Apply H_k to every column of W
                for (std::size_t j = 0; j < rows; ++j) {

                    // dot = v_k^T W[:, j]
                    double dot = W[j * cols + k];

                    for (std::size_t i = k + 1; i < cols; ++i) {
                        dot += At[k * cols + i] * W[j * cols + i];
                    }

                    // W[:,j] -= tau[k] * v_k * dot
                    const double scale = tau[k] * dot;

                    W[j * cols + k] -= scale;

                    for (std::size_t i = k + 1; i < cols; ++i) {
                        W[j * cols + i] -= scale * At[k * cols + i];
                    }
                }
            }

            return SVDResult{U_A, singular_values, W};
        }

}