#include <limits>
#include <vector>
#include <span>
#include <cmath>
#include <stdexcept>
#include "detail/stable_norm.hpp"


namespace hdos::detail{
double householder_reflector(std::span<double> x){
    std::size_t n = x.size();

    if (n==0){
        throw std::invalid_argument("Invalid input");
    }

    for (double entry : x){
        if (std::isnan(entry) || std::isinf(entry)){
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
    double scale_count = 0.0;

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
}