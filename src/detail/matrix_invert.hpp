#include <cmath>
#include <iostream>
#include "cholesky.hpp"
#include "triangular_solver.hpp"


using namespace std;


//Transpose matrix
vector<double> transpose(const vector<double>& M, size_t rows, size_t cols){
    vector<double> M_adj(rows * cols);
    for (size_t k = 0; k < rows; ++k){
        for (size_t j = 0; j < cols; ++j){
            M_adj[k*cols + j] = M[k + j*rows];    
        }
    }
    return M_adj;
}

//Solve positive definite from cholesky decomposition
vector<double> solve_pos_definite_cholesky(const vector<double>& L, const vector<double>& b){
    // (LL*)x = b iff L(L*x) = b
    const size_t N = static_cast<size_t>(sqrt(L.size()));    
    const vector<double> y = lower_triangular_solver(L, b);
    const vector<double> L_adj = transpose(L, N, N);
    const vector<double> x = upper_triangular_solver(L_adj, y);
    return x;
}

//Invert
vector<double> invert_pos_def_cholesky(const vector<double>& L){
       const size_t N = static_cast<size_t>(sqrt(L.size()));
       vector<double> L_inv(L.size());
       vector<double> e_k(N, 0.0);
       e_k[0] = 1.0;

       for (size_t k=0; k<N; ++k){ 
        const vector<double> L_inv_k_col = solve_pos_definite_cholesky(L, e_k);
        copy(L_inv_k_col.begin(),
             L_inv_k_col.end(),
             L_inv.begin() + k*N);

        if (k < N-1){
            e_k[k] = 0.0;
            e_k[k+1] = 1.0;
        }
       } 
       return L_inv;
}

//Invert a general positive definite matrix
vector<double> invert_pos_def(const vector<double>& M){
    const vector<double> L = cholesky_decomp(M);
    return invert_pos_def_cholesky(L);
}


// int main(){
//     vector<double> M = {2,1,1,2};
//     vector <double> L =  invert_pos_def(M);
//     mat_print(L);
//     M = {4,2,2,3};
//     L =  invert_pos_def(M);
//     mat_print(L);
//     M = {2,1,0, 1,2, 1, 0, 1,2};
//     L =  invert_pos_def(M);
//     mat_print(L);
//     return 0;
// }