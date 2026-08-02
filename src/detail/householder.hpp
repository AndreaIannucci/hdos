#pragma once

#include <span>

namespace hdos::detail{
double householder_reflector(std::span<double> x);
}