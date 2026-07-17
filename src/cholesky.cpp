#include<vector>
#include <cmath>
#include <iostream>
#include <format>

using namespace std;


// TODO:
// Change double to template
// Maintain "triangular" instead of complete matrix 
// Check size of x is compatible with L



vector<double> cholesky_decomp(const vector<double>& M){
    const size_t N = static_cast<size_t>(sqrt(M.size()));
    vector<double>L(M.size()); 
    for (size_t k =0; k < N; ++k){
        for (size_t j =0; j <= k; ++j){
            double sum = 0;
            for (size_t i = 0; i < j; ++i){
                sum += L[k  + i*N] * L[j + N * i];
            }
            
            if (k == j){
                double addend = M[k + N*k] - sum;
                if (addend < 0){
                    throw invalid_argument("The matrix is not positive definite");
                }
                L[k + N*j] = sqrt(addend); 
            }
            else{
                double denom = L[j + N*j];
                if (denom == 0){
                    throw invalid_argument("The matrix is not positive definite");
                }
                L[k + N*j] = (1.0 / denom) * (M[k + j*N] - sum);
            
            }
        }
    }
    return L;
}

void rk1_cholesky(vector<double>& L, vector<double> x){
    // Rank one update of the triangular cholesky L by x

    double c = 1;
    auto N = x.size();
    for (size_t k = 0; k < N-1; k++){
        auto start = L.begin() + N* k;
        auto end = L.begin() + N* (k+1);
        vector<double> l = vector<double>(start, end);

        auto lk = l[k];
        auto xk = x[k];
        auto dk = std::sqrt(lk*lk + c*xk*xk);
        
        for (auto j = 0; j < N; j++){
            L[k + j*N] = (lk/dk)*l[j]+(c* xk/dk)*x[j];
        }
        for (auto j = 0; j < N; j++){
            x[j] = x[j]-l[j]*(xk/lk);
        }
        
        c = c * (lk / dk)*(lk / dk);
    L[(N-1) + N*(N-1)] = sqrt(L[(N-1) + N*(N-1)]*L[(N-1) + N*(N-1)] + c*x[N-1]*x[N-1]);
    }
}


void mat_print(const vector<double>& L){
    size_t N = (int) std::sqrt(L.size());
    for (size_t k=0; k < N; k ++){
        for (size_t j=0; j < N; j ++){
            cout << std::format("{}", L[k + j*N]) <<  " ";
        }
        cout << "\n";
    }
}

// int main(){
//     vector<double> v = {1,2,3,5};
//     vector<double> x = {1,0};
//     vector<double> L = cholesky(v);
//     mat_print(L);
//     return 0;
// }