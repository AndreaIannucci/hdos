
#include <cmath>
#include <span>
#include <stdexcept>

#include "detail/jacobi_rotation.hpp"

namespace hdos::detail{
JacobiRotation jacobi_rotation(double alpha, double  beta, double gamma){
    if (gamma == 0){
        return JacobiRotation{1.0, 0.0};
    }
    double zeta = 0.5 * (beta - alpha) / gamma;
    double t = std::copysign(1.0, zeta)/
        (std::abs(zeta) + std::hypot(1, zeta));
    double cosine = 1 / std::hypot(1, t);
    double sine = t * cosine;

    return JacobiRotation{cosine, sine};
}


void apply_jacobi_rotation(std::span<double> x,std::span<double> y, double cosine, double sine){

    if (x.size() != y.size()){
        throw std::invalid_argument("x and y must have same size");
    }

    for (std::size_t i = 0; i<x.size(); ++i){
        double old_x = x[i];
        double old_y = y[i];

        x[i] = cosine * old_x
             - sine   * old_y;

        y[i] = sine   * old_x
             + cosine * old_y;
    }
}
}