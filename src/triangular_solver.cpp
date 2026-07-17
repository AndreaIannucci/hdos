#include <cmath>
#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

// Implement a lower triangular solver
// @params 
//      L: column major matrix
//      b: intercept     
vector<double> lower_triangular_solver(const vector<double>& L, const vector<double>& b){
    const size_t N = b.size();

    vector<double> solution(N);

    for (size_t k = 0; k< N; ++k){
        auto right_side = b[k];
        for (size_t j = 0; j < k; ++j){
            right_side -= L[k + j*N] * solution[j];
        }
        auto diag = L[k + k*N];
        if (diag == 0.0) {
            throw std::invalid_argument("The system is singular");
        }

        solution[k] = right_side / diag;
    } 
    return solution;
}

// Implement a upper triangular solver
// @params 
//      L: column major matrix
//      b: intercept     
vector<double> upper_triangular_solver(const vector<double>& L, const vector<double>& b){
    const size_t N = b.size();

    vector<double> solution(N);
    
    for (size_t k = 0; k< N; ++k){
        auto k_rev = N-1-k; 
        auto right_side = b[k_rev];
        for (size_t j = 0; j < k; ++j){
            auto j_rev = N-1-j;
            right_side -= L[k_rev + j_rev*N] * solution[j_rev];
        }
        auto diag = L[k_rev + k_rev*N];
        if (diag == 0.0) {
            throw std::invalid_argument("The system is singular");
        }

        solution[k_rev] = right_side / diag;
    } 
    return solution;
}

// int main(){
//     vector<double> L1 = {1,0,2,1};
//     vector<double> b = {4,3};
//     vector<double> sol = upper_triangular(L1, b);
//     for (double el:sol){
//         cout << el << " ";
//     }
//     return 0;
// }
