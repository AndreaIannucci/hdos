#include<vector>
#include<algorithm>
#include<span>
#include <stdexcept>
#include <cmath>
#include "detail/QR.hpp"
#include "detail/householder.hpp"


namespace hdos::detail{
std::vector<double> QR_decomp(
    std::span<double> A,
    std::size_t rows,
    std::size_t cols
)
{
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument(
            "The matrix cannot have 0 rows or 0 columns"
        );
    }

    if (
        cols >
            std::numeric_limits<std::size_t>::max() / rows ||
        A.size() != rows * cols
    ) {
        throw std::invalid_argument(
            "Invalid matrix dimensions"
        );
    }

    for (double value : A) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "The matrix must contain only finite entries"
            );
        }
    }

    std::size_t reflector_count = std::min(rows, cols);
    std::vector<double> tau(reflector_count, 0.0);

    for (std::size_t k=0; k<reflector_count; ++k){
        std::size_t reflector_length = rows - k;
        std::size_t reflector_start = k * rows + k;

        std::span<double> packed =
         A.subspan(reflector_start, reflector_length);
        
        tau[k]  = detail::householder_reflector(packed);

        if (k+1 < cols){
            detail::apply_householder_left(A,
                rows,
                cols,
                k,
                k + 1,
                packed,
                tau[k]
            );
        }
    } 
    return tau;  
 }


std::vector<double> extract_R(std::span<const double> packed_qr, std::size_t rows, std::size_t cols){
    if (rows == 0 || cols == 0){
        throw std::invalid_argument("The matrix cannot have 0 rows or 0 columns");
    }
    if (rows * cols != packed_qr.size()){
        throw std::invalid_argument("Invalid matrix dimensions");
    }

    std::size_t reduced_rows = std::min(rows, cols);
    std::vector<double> R(reduced_rows * cols, 0.0);

    for (std::size_t k = 0; k<cols ; ++k){
        std::size_t entries_to_copy = std::min(k +1, reduced_rows);

        for (std::size_t j = 0; j<entries_to_copy ; ++j){
            std::size_t packed_index = j + k *rows;
            std::size_t R_index = j + k * reduced_rows;

            R[R_index] = packed_qr[packed_index];
        }
    }
    
    return R;
}


void apply_Q_transpose(
    std::span<const double> packed_qr,
    std::span<const double> tau,
    std::size_t qr_rows,
    std::size_t qr_cols,
    std::span<double> B,
    std::size_t B_cols){

        if (qr_rows == 0 || qr_cols == 0){
        throw std::invalid_argument("The matrix cannot have 0 rows or 0 columns");
        }

        if( packed_qr.size() != qr_rows * qr_cols){
            throw std::invalid_argument("Invalid matrix dimension");
        }

        if( B.size() != qr_rows * B_cols){
            throw std::invalid_argument("Invalid matrix dimension");
        }

        std::size_t reflector_count = std::min(qr_rows, qr_cols);

        if (tau.size() != reflector_count){
            throw std::invalid_argument("Invalid reflector size");
        }

        for (std::size_t k=0; k< reflector_count; ++k){
            std::size_t reflector_start = k*qr_rows + k;
            std::size_t reflector_length = qr_rows - k;
            std::span<const double> packed_reflector = packed_qr.subspan(reflector_start, reflector_length);
            detail::apply_householder_left(
            B,
            qr_rows,
            B_cols,
            k,
            0,
            packed_reflector,
            tau[k]
        );
        }
    }


    void apply_Q(
    std::span<const double> packed_qr,
    std::span<const double> tau,
    std::size_t qr_rows,
    std::size_t qr_cols,
    std::span<double> B,
    std::size_t B_cols){

        if (qr_rows == 0 || qr_cols == 0){
        throw std::invalid_argument("The matrix cannot have 0 rows or 0 columns");
        }

        if( packed_qr.size() != qr_rows * qr_cols){
            throw std::invalid_argument("Invalid matrix dimension");
        }

        if( B.size() != qr_rows * B_cols){
            throw std::invalid_argument("Invalid matrix dimension");
        }

        std::size_t reflector_count = std::min(qr_rows, qr_cols);

        if (tau.size() != reflector_count){
            throw std::invalid_argument("Invalid reflector size");
        }

        for (std::size_t k = reflector_count; k-- > 0;) {

            std::size_t reflector_start = k * qr_rows + k;
            std::size_t reflector_length = qr_rows - k;

            std::span<const double> packed_reflector = packed_qr.subspan(reflector_start, reflector_length);
            detail::apply_householder_left(
            B,
            qr_rows,
            B_cols,
            k,
            0,
            packed_reflector,
            tau[k]
        );
        }
    }

    std::vector<double> form_Q(
    std::span<const double> packed_qr,
    std::span<const double> tau,
    std::size_t rows,
    std::size_t cols){

        if (rows == 0 || cols == 0){
        throw std::invalid_argument("The matrix cannot have 0 rows or 0 columns");
        }

        if( packed_qr.size() != rows * cols){
            throw std::invalid_argument("Invalid matrix dimension");
        }

        std::size_t reflector_count = std::min(rows, cols);

        if (tau.size() != reflector_count){
            throw std::invalid_argument("Invalid reflector size");
        }

        std::vector<double> Q(rows  * reflector_count, 0.0);
        for (std::size_t k =0; k<reflector_count; ++k){
            Q[k + k*rows] = 1;
        }

        apply_Q(
            packed_qr,
            tau,
            rows,
            cols,
            Q,
            reflector_count);
        
        return Q;
    }
}
