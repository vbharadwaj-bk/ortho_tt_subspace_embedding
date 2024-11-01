#pragma once

#include <iostream>
#include <string>
#include <random>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "black_box_tensor.hpp"
#include "sparse_tensor.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) LowRankTensor : public BlackBoxTensor {
public:
    unique_ptr<NPBufferList<double>> U_py_bufs;
    vector<Buffer<double>> &U;
    uint32_t N;
    uint64_t R;

    Buffer<double> sigma;
    Buffer<double> col_norms;

    bool is_static;
    double normsq;

    LowRankTensor(uint64_t R, uint64_t max_rhs_rows, 
        py::list U_py
        )
    :
    U_py_bufs(new NPBufferList<double>(U_py)),
    U(U_py_bufs->buffers),
    sigma({R}),
    col_norms({(uint64_t) U_py_bufs->length, R})
    {
        this->max_rhs_rows = max_rhs_rows;
        this->R = R;
        this->N = U_py_bufs->length;

        std::fill(sigma(), sigma(R), 1.0);
        std::fill(col_norms(), col_norms(N * R), 1.0);
        for(uint32_t i = 0; i < N; i++) {
            dims.push_back(U_py_bufs->buffers[i].shape[0]);
        }

        // This is a static tensor. We will compute and store the norm^2
        is_static = true;

        get_sigma(sigma, -1);
        normsq = ATB_chain_prod_sum(U, U, sigma, sigma);
    }

    LowRankTensor(uint64_t R, py::list U_py)
    :
    U_py_bufs(new NPBufferList<double>(U_py)),
    U(U_py_bufs->buffers),
    sigma({R}),
    col_norms({(uint64_t) U_py_bufs->length, R})
    {
        this->R = R;
        this->N = U_py_bufs->length;
        std::fill(sigma(), sigma(R), 1.0);
        std::fill(col_norms(), col_norms(N * R), 1.0);
        for(uint32_t i = 0; i < N; i++) {
            dims.push_back(U_py_bufs->buffers[i].shape[0]);
        }

        is_static = false;
    }

    double get_normsq() {
        if (! is_static) { 
            get_sigma(sigma, -1); 
            normsq = ATB_chain_prod_sum(U, U, sigma, sigma);
        }
        return normsq; 
    };

    double compute_residual_normsq(Buffer<double> &sigma_other, vector<Buffer<double>> &U_other) {
        get_sigma(sigma, -1); 
        double self_normsq = get_normsq();
        double other_normsq = ATB_chain_prod_sum(U_other, U_other, sigma_other, sigma_other);
        double inner_prod = ATB_chain_prod_sum(U_other, U, sigma_other, sigma);

        /*cout << self_normsq << " "
            << other_normsq << " "
            << inner_prod << " Data" << endl;*/

        //cout << self_normsq + other_normsq << " "
        //    << 2 * inner_prod << " Data" << endl;

        /*for(uint64_t i = 0; i < R; i++) {
            cout << sigma[i] << " ";
        }
        cout << endl;*/

        return max(self_normsq + other_normsq - 2 * inner_prod, 0.0);
    } 

    /*
    * Fills rhs_buf with an evaluation of the tensor starting from the
    * specified index in the array of samples. 
    */
    void materialize_rhs(Buffer<uint64_t> &samples_transpose, uint64_t j, Buffer<double> &rhs_buf) {
        get_sigma(sigma, -1);

        Buffer<double> partial_eval({samples_transpose.shape[0], R});

        #pragma omp parallel for 
        for(uint32_t i = 0; i < partial_eval.shape[0]; i++) {
            std::copy(sigma(), sigma(R), partial_eval(i * R));
            for(uint32_t k = 0; k < N; k++) {
                if(k != j) {
                    for(uint32_t u = 0; u < R; u++) {
                        partial_eval[i * R + u] *= U_py_bufs->buffers[k][samples_transpose[i * N + k] * R + u];
                    }
                } 
            }
        }

        if(j >= N) {
            if(rhs_buf.shape[0] != partial_eval.shape[0] || rhs_buf.shape[1] != partial_eval.shape[1]) {
                cout << "Shape mismatch, terminating!" << endl;
                exit(1);
            }
            else {
                std::copy(partial_eval(), partial_eval(samples_transpose.shape[0] * R), rhs_buf());
            }
        }
        else {
            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                partial_eval.shape[0],
                rhs_buf.shape[1],
                R,
                1.0,
                partial_eval(),
                R,
                U_py_bufs->buffers[j](),
                R,
                0.0,
                rhs_buf(),
                rhs_buf.shape[1]
            );
        }
    }

    void materialize_rhs_py(py::array_t<uint64_t> &samples_py, uint64_t j, py::array_t<double> &rhs_buf_py) {
        Buffer<uint64_t> samples(samples_py);
        Buffer<double> rhs_buf(rhs_buf_py);
        materialize_rhs(samples, j, rhs_buf);
    }

    void preprocess(Buffer<uint64_t> &samples_transpose, uint64_t j) {

    }

    void execute_exact_mttkrp(vector<Buffer<double>> &U_L, uint64_t j, Buffer<double> &mttkrp_res) {
        uint64_t R_L = U_L[0].shape[1];
        Buffer<double> sigma({R});
        Buffer<double> chain_had_prod({R, R_L});
        get_sigma(sigma, -1);

        Buffer<double> ones({R_L});
        std::fill(ones(), ones(R_L), 1.0);

        ATB_chain_prod(
                U,
                U_L,
                sigma,
                ones,
                chain_had_prod,
                j);

        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            (uint32_t) U[j].shape[0],
            (uint32_t) U_L[j].shape[1],
            (uint32_t) R,
            1.0,
            U[j](),
            (uint32_t) R,
            chain_had_prod(),
            R_L,
            0.0,
            mttkrp_res(),
            R_L);
    }

    /*
    * Returns the product of all column norms, except for the
    * one specified by the parameter. If the parameter is -1,
    * the product of all column norms is returned. 
    */
    void get_sigma(Buffer<double> &sigma_out, int j) {
        std::copy(sigma(), sigma(R), sigma_out());
    }

    void get_sigma_py(py::array_t<double> sigma_py_out, int j) {
        Buffer<double> sigma_py(sigma_py_out);
        get_sigma(sigma_py, j);
    }

    /*void multiply_by_random_entries() {
        for(uint32_t i = 0; i < N; i++) {

        }
    }*/

    // Pass j = -1 to renormalize all factor matrices.
    // The array sigma is always updated 
    void renormalize_columns(int j) {
        for(int i = 0; i < (int) N; i++) {
            if(j == -1 || j == i) {
                std::fill(col_norms(i * R), 
                    col_norms((i + 1) * R), 0.0);

                #pragma omp parallel
{
                Buffer<double> thread_loc_norms({R});
                std::fill(thread_loc_norms(), 
                    thread_loc_norms(R), 0.0);

                #pragma omp for 
                for(uint32_t u = 0; u < dims[i]; u++) {
                    for(uint32_t v = 0; v < R; v++) {
                        double entry = U[i][u * R + v];
                        thread_loc_norms[v] += entry * entry; 
                    }
                }

                for(uint32_t v = 0; v < R; v++) {
                    #pragma omp atomic 
                    col_norms[i * R + v] += thread_loc_norms[v]; 
                }
}

                for(uint32_t v = 0; v < R; v++) { 
                    col_norms[i * R + v] = sqrt(col_norms[i * R + v]);
                }


                for(uint32_t v = 0; v < R; v++) {
                    if(col_norms[i * R + v] <= 1e-7) {
                        col_norms[i * R + v] = 1.0;
                    }
                }

                #pragma omp parallel for
                for(uint32_t u = 0; u < dims[i]; u++) {
                    for(uint32_t v = 0; v < R; v++) {
                        U[i][u * R + v] /= col_norms[i * R + v]; 
                    }
                }
            }
        }

        if(j == -1) {
            std::fill(sigma(), sigma(R), 1.0);
            for(int i = 0; i < (int) N; i++) {
                for(uint32_t v = 0; v < R; v++) {
                    sigma[v] *= col_norms[i * R + v];
                }
            }
        }
        else {
            std::copy(col_norms(j * R), col_norms((j+1) * R), sigma());
        }
    }

    void initialize_rrf(SparseTensor &sp_ten) {
        sp_ten.execute_rrf(U);
        renormalize_columns(-1);
    }

    // This function is just for testing purposes
    void multiply_random_factor_entries(double rho, double A) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution coin(rho);
        for(int j = 0; j < (int) N; j++) {
            uint64_t Ij = U[j].shape[0];
            for(uint64_t i = 0; i < Ij; i++) {
                for(uint64_t k = 0; k < R; k++) {
                    if(coin(gen)) {
                        U[j][i * R + k] *= A;
                    }
                }
            }
        }
    }
};