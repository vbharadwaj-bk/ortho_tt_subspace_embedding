#pragma once

#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <initializer_list>
#include <chrono>
#include "omp.h"
#include "cblas.h"
#include "lapacke.h"

using namespace std;
namespace py = pybind11;

inline uint32_t divide_and_roundup(uint32_t n, uint32_t m) {
    return (n + m - 1) / m;
}

inline void log2_round_down(uint32_t m, 
        uint32_t& log2_res, 
        uint32_t& lowest_power_2) {
    
    assert(m > 0);
    log2_res = 0;
    lowest_power_2 = 1;

    while(lowest_power_2 * 2 <= m) {
        log2_res++; 
        lowest_power_2 *= 2;
    }
}

//#pragma GCC visibility push(hidden)
template<typename T>
class NumpyArray {
public:
    py::buffer_info info;
    T* ptr;

    NumpyArray(py::array_t<T> arr_py) {
        info = arr_py.request();
        ptr = static_cast<T*>(info.ptr);
    }
};

template<typename T>
class __attribute__((visibility("hidden"))) Buffer {
    py::buffer_info info;
    unique_ptr<T[]> managed_ptr;
    T* ptr;
    uint64_t dim0;
    uint64_t dim1;
    bool initialized;

public:
    vector<uint64_t> shape;

    Buffer(Buffer&& other)
        :   info(std::move(other.info)), 
            managed_ptr(std::move(other.managed_ptr)),
            ptr(std::move(other.ptr)),
            dim0(other.dim0),
            dim1(other.dim1),
            initialized(other.initialized),
            shape(std::move(other.shape))
    {}
    Buffer& operator=(const Buffer& other) = default;

    void steal_resources(Buffer& other) {
        info = std::move(other.info); 
        managed_ptr = std::move(other.managed_ptr);
        ptr = other.ptr;
        dim0 = other.dim0;
        dim1 = other.dim1;
        shape = other.shape;
        initialized = other.initialized;
    }

    Buffer(py::array_t<T> arr_py, bool copy) {
        info = arr_py.request();

        if(info.ndim == 2) {
            dim0 = info.shape[0];
            dim1 = info.shape[1];
        }
        else if(info.ndim == 1) {
            dim0 = info.shape[0];
            dim1 = 1;
        }

        uint64_t buffer_size = 1;
        for(int64_t i = 0; i < info.ndim; i++) {
            shape.push_back(info.shape[i]);
            buffer_size *= info.shape[i];
        }

        if(! copy) {
            ptr = static_cast<T*>(info.ptr);
        }
        else {
            managed_ptr.reset(new T[buffer_size]);
            ptr = managed_ptr.get();
            std::copy(static_cast<T*>(info.ptr), static_cast<T*>(info.ptr) + info.size, ptr);
        }
        initialized = true;
    }

    Buffer(py::array_t<T> arr_py) :
        Buffer(arr_py, false)
    {
        // Default behavior is a thin alias of the C++ array 
    }

    Buffer(initializer_list<uint64_t> args) {
        initialized = false;
        reset_to_shape(args);
    }

    Buffer(initializer_list<uint64_t> args, T* ptr) {
        for(uint64_t i : args) {
            shape.push_back(i);
        }

        if(args.size() == 2) {
            dim0 = shape[0];
            dim1 = shape[1];
        }

        this->ptr = ptr;
        initialized = true;
    }

    Buffer() {
        initialized = false;
    }

    void reset_to_shape(initializer_list<uint64_t> args) {
        shape.clear();
        uint64_t buffer_size = 1;
        for(uint64_t i : args) {
            buffer_size *= i;
            shape.push_back(i);
        }

        if(args.size() == 2) {
            dim0 = shape[0];
            dim1 = shape[1];
        }

        managed_ptr.reset(new T[buffer_size]);
        ptr = managed_ptr.get();
        initialized = true;
    }

    T* operator()() {
        return ptr;
    }

    T* operator()(uint64_t offset) {
        return ptr + offset;
    }

    // Assumes that this array is a row-major matrix 
    T* operator()(uint64_t off_x, uint64_t off_y) {
        return ptr + (dim1 * off_x) + off_y;
    }

    T& operator[](uint64_t offset) {
        return ptr[offset];
    }

    void print() {
        cout << "------------------------" << endl;
        if(shape.size() == 1) {
            cout << "[ " << " "; 
            for(uint64_t i = 0; i < shape[0]; i++) {
                cout << ptr[i] << " ";
            }
            cout << "]" << endl;
            return;
        }
        else if(shape.size() == 2) {
            for(uint64_t i = 0; i < shape[0]; i++) {
                cout << "[ ";
                for(uint64_t j = 0; j < shape[1]; j++) {
                    cout << ptr[i * shape[1] + j] << " ";
                }
                cout << "]" << endl; 
            }
        }
        else {
            cout << "Cannot print buffer with shape: ";
            for(uint64_t i : shape) {
                cout << i << " ";
            }
            cout << endl;
        }
        cout << "------------------------" << endl;
    }

    ~Buffer() {}
};

template<typename T>
class __attribute__((visibility("hidden"))) NPBufferList {
public:
    vector<Buffer<T>> buffers;
    int length;

    NPBufferList(py::list input_list) {
        length = py::len(input_list);
        for(int i = 0; i < length; i++) {
            py::array_t<T> casted = input_list[i].cast<py::array_t<T>>();
            buffers.emplace_back(casted);
        }
    }
};

/*
* exclude is the index of a matrix to exclude from the chain Hadamard product. Pass -1
* to include all components in the chain Hadamard product.
*/
void ATB_chain_prod(
        vector<Buffer<double>> &A,
        vector<Buffer<double>> &B,
        Buffer<double> &sigma_A, 
        Buffer<double> &sigma_B,
        Buffer<double> &result,
        int exclude) {

        uint64_t N = A.size();
        uint64_t R_A = A[0].shape[1];
        uint64_t R_B = B[0].shape[1];

        vector<unique_ptr<Buffer<double>>> ATB;
        for(uint64_t i = 0; i < A.size(); i++) {
                ATB.emplace_back();
                ATB[i].reset(new Buffer<double>({R_A, R_B}));
        }

        for(uint64_t i = 0; i < R_A; i++) {
                for(uint64_t j = 0; j < R_B; j++) {
                        result[i * R_B + j] = sigma_A[i] * sigma_B[j];
                }

        }

        // Can replace with a batch DGEMM call
        for(uint64_t i = 0; i < N; i++) {
            if(((int) i) != exclude) {
                uint64_t K = A[i].shape[0];
                cblas_dgemm(
                        CblasRowMajor,
                        CblasTrans,
                        CblasNoTrans,
                        R_A,
                        R_B,
                        K,
                        1.0,
                        A[i](),
                        R_A,
                        B[i](),
                        R_B,
                        0.0,
                        (*(ATB[i]))(),
                        R_B
                );
            }
        }

        #pragma omp parallel 
{
        for(uint64_t k = 0; k < N; k++) {
                if(((int) k) != exclude) {
                    #pragma omp for collapse(2)
                    for(uint64_t i = 0; i < R_A; i++) {
                            for(uint64_t j = 0; j < R_B; j++) {
                                    result[i * R_B + j] *= (*(ATB[k]))[i * R_B + j];
                            }
                    }
                }
        }
}
}

double ATB_chain_prod_sum(
        vector<Buffer<double>> &A,
        vector<Buffer<double>> &B,
        Buffer<double> &sigma_A, 
        Buffer<double> &sigma_B) {

    uint64_t R_A = A[0].shape[1];
    uint64_t R_B = B[0].shape[1];
    Buffer<double> result({R_A, R_B});
    ATB_chain_prod(A, B, sigma_A, sigma_B, result, -1);
    return std::accumulate(result(), result(R_A * R_B), 0.0); 
}

void compute_pinv_square(Buffer<double> &M, Buffer<double> &out, uint64_t target_rank) {
    uint64_t R = M.shape[0];
    double eigenvalue_tolerance = 1e-11;
    Buffer<double> lambda({R});

    LAPACKE_dsyev( CblasRowMajor, 
                    'V', 
                    'U', 
                    R,
                    M(), 
                    R, 
                    lambda() );

    //cout << "Lambda: ";
    for(uint32_t v = 0; v < R; v++) {
        //cout << lambda[v] << " ";
        if(v >= R - target_rank && lambda[v] > eigenvalue_tolerance) {
            for(uint32_t u = 0; u < R; u++) {
                M[u * R + v] = M[u * R + v] / sqrt(lambda[v]); 
            }
        }
        else {
            for(uint32_t u = 0; u < R; u++) {
                M[u * R + v] = 0.0; 
            }
        }
    }
    //cout << "]" << endl;

    cblas_dsyrk(CblasRowMajor, 
                CblasUpper, 
                CblasNoTrans,
                R,
                R, 
                1.0, 
                (const double*) M(), 
                R, 
                0.0, 
                out(), 
                R);

}

void compute_pinv(Buffer<double> &in, Buffer<double> &out) {
    uint64_t R = in.shape[1];
    Buffer<double> M({R, R});

    std::fill(M(), M(R * R), 0.0);

    uint64_t I = in.shape[0];
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        uint64_t work = (I + num_threads - 1) / num_threads;
        uint64_t start = min(work * thread_id, I);
        uint64_t end = min(work * (thread_id + 1), I);

        if(end - start > 0) {
            Buffer<double> local({R, R});
            cblas_dsyrk(CblasRowMajor, 
                        CblasUpper, 
                        CblasTrans,
                        R,
                        end-start, 
                        1.0, 
                        in(start * R), 
                        R, 
                        0.0, 
                        local(), 
                        R);

            for(uint64_t i = 0; i < R * R; i++) {
                #pragma omp atomic
                M[i] += local[i];
            }
        }

    }

    // Compute pseudo-inverse of the input matrix through dsyrk and eigendecomposition  
    /*cblas_dsyrk(CblasRowMajor, 
                CblasUpper, 
                CblasTrans,
                R,
                in.shape[0], 
                1.0, 
                in(), 
                R, 
                0.0, 
                M(), 
                R);*/

    uint64_t target_rank = min(in.shape[0], R);
    compute_pinv_square(M, out, target_rank);
}

typedef chrono::time_point<std::chrono::steady_clock> my_timer_t; 

my_timer_t start_clock() {
    return std::chrono::steady_clock::now();
}

double stop_clock_get_elapsed(my_timer_t &start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}