#include <limits>
#include <vector>
#include <span>
#include <cmath>
#include <stdexcept>
#include "detail/stable_norm.hpp"
#include "detail/householder.hpp"


namespace hdos::detail{
double householder_reflector(std::span<double> x){
    std::size_t n = x.size();

    if (n==0){
        throw std::invalid_argument("Invalid input");
    }

    for (double entry : x){
        if (! std::isfinite(entry)){
            throw std::invalid_argument("No entry can be nan or infty");
        }
    }

    if (n==1){
        return 0.0;
    }

    double alpha = x[0];
    double x_norm = stable_norm(x.subspan(1,x.size()-1));

    if (x_norm == 0){
        return 0.0;
    }

    double beta = - std::copysign(std::hypot(alpha, x_norm), alpha);

    const double safe_minimum = std::numeric_limits<double>::min() / std::numeric_limits<double>::epsilon();
    const double inverse_safe_minimum = 1.0 / safe_minimum;
    std::size_t scale_count = 0.0;

    if (beta < safe_minimum && beta > -safe_minimum){
        while (beta < safe_minimum && beta > -safe_minimum){
            scale_count += 1;
            alpha *= inverse_safe_minimum;
            for (std::size_t j =1; j < n; ++j){
                x[j] *= inverse_safe_minimum;
            } 
            beta *= inverse_safe_minimum;
        }
        x_norm = detail::stable_norm(x.subspan(1,x.size()-1));
        beta = - std::copysign(std::hypot(alpha, x_norm), alpha);
    }
    double tau = (beta - alpha) / beta;
    double denominator = (alpha - beta);

    for (std::size_t j =1; j < n; ++j){
                x[j] /= denominator;
            }

    for (std::size_t j= 0; j < scale_count; ++j){
        beta =  beta * safe_minimum;
    }
    x[0] = beta;
    return tau;
}

void apply_householder_left(
    std::span<double> A,
    std::size_t rows,
    std::size_t cols,
    std::size_t start_row,
    std::size_t start_col,
    std::span<const double> packed,
    double tau)
{
    if (rows == 0) {
        throw std::invalid_argument(
            "The matrix cannot have zero rows");
    }

    if (A.size() != rows * cols) {
        throw std::invalid_argument(
            "Invalid matrix dimensions");
    }

    if (start_row > rows || start_col > cols) {
        throw std::invalid_argument(
            "Invalid block starting position");
    }

    const std::size_t block_rows = rows - start_row;

    if (packed.size() != block_rows) {
        throw std::invalid_argument(
            "Invalid reflector dimension");
    }

    if (block_rows == 0 || start_col == cols || tau == 0.0) {
        return;
    }

    for (std::size_t column = start_col;
         column < cols;
         ++column)
    {
        const std::size_t first_index =
            column * rows + start_row;

        // The first component of v is implicitly 1.
        // packed[0] stores beta and is not used here.
        double dot = A[first_index];

        for (std::size_t local_row = 1;
             local_row < block_rows;
             ++local_row)
        {
            dot +=
                packed[local_row] *
                A[first_index + local_row];
        }

        const double coefficient = tau * dot;

        A[first_index] -= coefficient;

        for (std::size_t local_row = 1;
             local_row < block_rows;
             ++local_row)
        {
            A[first_index + local_row] -=
                packed[local_row] * coefficient;
        }
    }
}



void apply_householder_right(
    std::span<double> A,
    std::size_t rows,
    std::size_t cols,
    std::size_t start_row,
    std::size_t start_col,
    std::span<const double> packed,
    double tau)
{
    if (cols == 0) {
        throw std::invalid_argument(
            "The matrix cannot have zero columns");
    }

    if (A.size() != rows * cols) {
        throw std::invalid_argument(
            "Invalid matrix dimensions");
    }

    if (start_row > rows || start_col > cols) {
        throw std::invalid_argument(
            "Invalid block starting position");
    }

    const std::size_t block_cols = cols - start_col;

    if (packed.size() != block_cols) {
        throw std::invalid_argument(
            "Invalid reflector dimension");
    }

    if (start_row == rows ||
        block_cols == 0 ||
        tau == 0.0)
    {
        return;
    }

    for (std::size_t row = start_row;
         row < rows;
         ++row)
    {
        // First element of the affected row block:
        // A[row, start_col].
        const std::size_t first_index =
            row + start_col * rows;

        // v[0] is implicitly 1.
        // packed[0] contains beta and is not used.
        double dot = A[first_index];

        for (std::size_t local_col = 1;
             local_col < block_cols;
             ++local_col)
        {
            const std::size_t matrix_index =
                row + (start_col + local_col) * rows;

            dot +=
                packed[local_col] *
                A[matrix_index];
        }

        const double coefficient = tau * dot;

        A[first_index] -= coefficient;

        for (std::size_t local_col = 1;
             local_col < block_cols;
             ++local_col)
        {
            const std::size_t matrix_index =
                row + (start_col + local_col) * rows;

            A[matrix_index] -=
                packed[local_col] * coefficient;
        }
    }
}

}