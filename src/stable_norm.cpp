#include "detail/stable_norm.hpp"

#include <vector>
#include <cmath>

using namespace std;


namespace hdos::detail{
// TODO infinite or nan elements
double stable_norm(std::span<const double> x){
    double scale = 0.0;
    double ssq = 1.0;

    for (double xi: x){
        if (xi != 0){
            double a = std::abs(xi);
            if (a > scale){
                ssq = 1 + ssq*(scale / a)* (scale / a);
                scale = a;
            }
            else{
                ssq += (a / scale) * (a / scale);
            }
        }
    }
    return scale * std::sqrt(ssq);
}
}