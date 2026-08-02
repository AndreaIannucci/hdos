#pragma once

#include <span>

namespace hdos::detail{
double stable_norm(std::span< const double> x);
}