#pragma once

#include<span>
#include <stdexcept>

namespace hdos::detail{
struct JacobiRotation {
    double cosine;
    double sine;
};

JacobiRotation jacobi_rotation(double alpha, double  beta, double gamma);
void apply_jacobi_rotation(std::span<double> x,std::span<double> y, double cosine, double sine);
}


