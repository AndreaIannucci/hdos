#include "detail/mat_print.hpp"
#include "detail/cholesky.hpp"

#include <vector>
#include <iostream>
#include <format>
#include <cmath>

namespace hdos::detail{
void mat_print(const std::vector<double>& L){
    size_t N = (int) std::sqrt(L.size());
    for (size_t k=0; k < N; k ++){
        for (size_t j=0; j < N; j ++){
            std::cout << std::format("{}", L[k + j*N]) <<  " ";
        }
        std::cout << "\n";
    }
}
}