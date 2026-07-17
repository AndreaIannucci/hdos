#include <vector>
#include <cmath>

using namespace std;


// TODO infinite or nan elements
double stable_norm(const vector<double>& x){
    double scale = 0.0;
    double ssq = 1.0;

    for (double xi: x){
        if (xi != 0){
            double a = abs(xi);
            if (a > scale){
                ssq = 1 + ssq*(scale / a)* (scale / a);
                scale = a;
            }
            else{
                ssq += (a / scale) * (a / scale);
            }
        }
    }
    return scale * sqrt(ssq);
}