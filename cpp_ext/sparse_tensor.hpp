#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "tensor.hpp"
#include "idx_lookup.hpp"
#include "sort_lookup.hpp"

using namespace std;
namespace py = pybind11;

/*
* By default, we assume this is a tensor with uint32_t index variables
* and double-precision values. 
*/
class __attribute__((visibility("hidden"))) SparseTensor : public Tensor {
public:
    Buffer<uint32_t> indices;
    Buffer<double> values;
    vector<unique_ptr<IdxLookup<uint32_t, double>>> lookups;

    uint64_t N, nnz;
    double normsq;

    SparseTensor(py::array_t<uint32_t> indices_py, 
        py::array_t<double> values_py,
        std::string method)
    :
    indices(indices_py),
    values(values_py)
    {
      nnz = indices.shape[0]; 
      N = indices.shape[1];

      for(uint64_t j = 0; j < N; j++) {
        if(method == "sort") {
          lookups.emplace_back(new SortIdxLookup(N, j, 
              indices(), 
              values(), 
              nnz));
        }
        else {
          cout << "Unknown lookup method passed, terminating..." << endl;
          exit(1);
        }
      }

      normsq = 0.0;
      #pragma omp parallel for reduction(+:normsq)
      for(uint64_t i = 0; i < nnz; i++) {
        normsq += values[i] * values[i];
      }

      // Sorting nonzeros would be good here... 
    }

    void execute_exact_mttkrp(vector<Buffer<double>> &U_L, uint64_t j, Buffer<double> &mttkrp_res) { 
      std::fill(mttkrp_res(), mttkrp_res(mttkrp_res.shape[0] * mttkrp_res.shape[1]), 0.0);
      lookups[j]->execute_exact_mttkrp(U_L, mttkrp_res);
    }

    void execute_rrf(vector<Buffer<double>> &U) {
      for(uint64_t j = 0; j < U.size(); j++) {
        uint64_t rows = U[j].shape[0];
        uint64_t cols = U[j].shape[1];
        std::fill(U[j](), U[j](rows * cols), 0.0);
        lookups[j]->execute_rrf(U[j]);
      }
    }

    void execute_downsampled_mttkrp(
            Buffer<uint64_t> &samples_transpose, 
            Buffer<double> &lhs,
            uint64_t j,
            Buffer<double> &result) {
      uint64_t J = samples_transpose.shape[0];
      uint64_t N = samples_transpose.shape[1];
      uint64_t Ij = result.shape[0];
      uint64_t R = result.shape[1];
      Buffer<uint32_t> samples_dcast({J, N});

      #pragma omp parallel for 
      for(uint64_t i = 0; i < J * N; i++) {
          samples_dcast[i] = (uint32_t) samples_transpose[i]; 
      } 

      // It is probably a good idea to sort and reweight samples here... 

      std::fill(result(), result(Ij * R), 0.0);
      lookups[j]->execute_spmm(samples_dcast, lhs, result);
    }

    double compute_residual_normsq(Buffer<double> &sigma, vector<Buffer<double>> &U) {
        return lookups[0]->compute_residual_normsq(sigma, U);
    }
 
    double get_normsq() {
      return normsq; 
    }
};