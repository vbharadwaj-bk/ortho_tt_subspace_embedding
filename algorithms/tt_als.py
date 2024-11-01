from tensors.tensor_train import *
from tensors.dense_tensor import *
from tensors.sparse_tensor import *

import time
import numpy as np

class TensorTrainALS:
    def __init__(self, ground_truth, tt_approx):
        self.ground_truth = ground_truth
        self.tt_approx = tt_approx

        print("Initialized TT-ALS!")

    def compute_exact_fit(self):
        '''
        This operation can be particularly expensive for dense tensors.
        '''
        ground_truth = self.ground_truth
        if isinstance(ground_truth, PyDenseTensor):
            tt_materialization = self.tt_approx.materialize_dense()
            return 1.0 - la.norm(tt_materialization - ground_truth.data) / ground_truth.data_norm
        elif isinstance(ground_truth, PySparseTensor):
            # Slow way to compute this, but okay for now
            # Also relies on core 0 being non-orthogonal 
            tt_values = self.tt_approx.evaluate_partial_fast(
                    ground_truth.tensor_idxs,
                    ground_truth.N, "left").squeeze()

            partial_diff_sum = np.sum(ground_truth.values * ground_truth.values - 2 * tt_values * ground_truth.values)
            tt_normsq = la.norm(self.tt_approx.U[0]) ** 2
            diff_norm = np.sqrt(np.maximum(tt_normsq + partial_diff_sum, 0.0))

            return 1.0 - (diff_norm / ground_truth.data_norm)
        else:
            raise NotImplementedError

    def execute_exact_als_sweeps_slow(self, num_sweeps):
        '''
        Assumes that the TT is in orthogonal
        form with core 0 non-orthogonal. This is a slow
        implementation mean for debugging.

        This is the single-site version of TT-ALS. 
        '''
        if not isinstance(self.ground_truth, PyDenseTensor):
            raise NotImplementedError

        tt_approx = self.tt_approx
        N = tt_approx.N

        def optimize_core(i):
            left_chain = tt_approx.left_chain_matricize(i)
            right_chain = tt_approx.right_chain_matricize(i)

            if len(left_chain.shape) == 0:
                left_cols = 1
            else:
                left_cols = left_chain.shape[1]

            if len(right_chain.shape) == 0:
                right_cols = 1
            else:
                right_cols = right_chain.shape[1] 

            design = np.kron(left_chain, right_chain)
            target_modes = list(range(N))
            target_modes.remove(i)
            target_modes.append(i)

            data_t = np.transpose(self.ground_truth.data, target_modes)
            data_mat = data_t.reshape([-1, data_t.shape[-1]])
            mode_size = data_mat.shape[1]
            tt_approx.U[i] = (design.T @ data_mat).reshape(left_cols, right_cols, mode_size).transpose([0, 2, 1]).copy()

        for _ in range(num_sweeps):
            for i in range(N - 1):
                optimize_core(i)
                tt_approx.orthogonalize_push_right(i)

            for i in range(N - 1, 0, -1):
                optimize_core(i)
                tt_approx.orthogonalize_push_left(i)

            print(tt_als.compute_exact_fit())

    def execute_exact_als_sweeps_sparse(self, num_sweeps, J, epoch_interval=5):
        print("Starting exact ALS for sparse tensors!")
        stats =         {   "als_time": 0.0, 
                            "fit_computation_time": 0.0,
                            "iterations": [],
                            "fits": [],
                            "als_iter_times": []
                            }
        tt_approx = self.tt_approx
        N = tt_approx.N

        # Evaluate the m4tc by processing batches of nonzero entries 
        B = 1000000

        def optimize_core(j):
            ground_truth = self.ground_truth
            lcols = tt_approx.U[j].shape[0]
            rcols = tt_approx.U[j].shape[2]
            result = np.zeros((tt_approx.dims[j], lcols * rcols), dtype=np.double)

            for i in range(0, ground_truth.nnz, B):
                lb = i
                ub = min(i+B, ground_truth.nnz)
                idx_batch = ground_truth.tensor_idxs[lb:ub].copy()
                values_batch = ground_truth.values[lb:ub].copy()

                left_rows = tt_approx.evaluate_partial_fast(idx_batch, j, "left")
                right_rows = tt_approx.evaluate_partial_fast(idx_batch, j, "right")
                tt_approx.internal_sampler.m4tc(idx_batch, j, left_rows, right_rows, values_batch, result)

            tt_approx.U[j] = result.reshape(tt_approx.dims[j], lcols, rcols).transpose([1, 0, 2]).copy()


        for i in range(num_sweeps):
            als_start = time.time()
            print(f"Starting sweep {i}...")
            for j in range(N - 1):
                optimize_core(j)
                tt_approx.orthogonalize_push_right(j)
                tt_approx.update_internal_sampler(j, "left", True)

            for j in range(N - 1, 0, -1):
                optimize_core(j)
                tt_approx.orthogonalize_push_left(j)
                tt_approx.update_internal_sampler(j, "right", True)

            tt_approx.update_internal_sampler(0, "left", False)
            stats["als_time"] += time.time() - als_start

            if i % epoch_interval == 0:
                fit_computation_start = time.time()
                stats["iterations"].append(i)
                fit = self.compute_exact_fit()
                stats["fits"].append(fit)
                stats["als_iter_times"].append(stats["als_time"]) 
                print(f"Fit after {i} iterations: {fit}")
                stats["fit_computation_time"] += time.time() - fit_computation_start 

        return stats 


    def execute_randomized_als_sweeps(self, num_sweeps, J, epoch_interval=5):
        print("Starting randomized ALS!")
        stats =         {   "als_time": 0.0, 
                            "fit_computation_time": 0.0,
                            "iterations": [],
                            "fits": [],
                            "als_iter_times": []
                            }
        tt_approx = self.tt_approx
        N = tt_approx.N
        def optimize_core(j):
            samples = np.zeros((J, N), dtype=np.uint64)
            left_rows = None
            right_rows = None
            if j > 0:
                left_samples = tt_approx.leverage_sample(j, J, "left")
                samples[:, :j] = left_samples

                left_rows = tt_approx.evaluate_partial_fast(samples, j, "left")
                left_cols = left_rows.shape[1]
            else:
                left_cols = 1
            if j < N - 1:
                right_samples = tt_approx.leverage_sample(j, J, "right")
                samples[:, j+1:] = right_samples
                
                right_rows = tt_approx.evaluate_partial_fast(samples, j, "right")
                right_cols = right_rows.shape[1]
            else:
                right_cols = 1

            if left_rows is None:
                design = right_rows
            elif right_rows is None:
                design = left_rows
            else:
                # Should probably write a custom kernel for this in C++ 
                design = np.einsum("ij,ik->ijk", left_rows, right_rows).reshape(J, -1)

            weights = la.norm(design, axis=1) ** 2 / design.shape[1] * J

            # We do this in two steps so we can (potentially) compute the
            # gram matrix 
            design = np.einsum("ij,i->ij", design, np.sqrt(1.0 / weights))
            design_gram = design.T @ design

            design = np.einsum("ij,i->ij", design, np.sqrt(1.0 / weights))

            result = np.zeros((tt_approx.dims[j], design.shape[1]), dtype=np.double)

            self.ground_truth.ten.execute_downsampled_mttkrp(
                    samples,
                    design,
                    j,
                    result)

            result = result @ la.pinv(design_gram) 
            tt_approx.U[j] = result.reshape(tt_approx.dims[j], left_cols, right_cols).transpose([1, 0, 2]).copy()


        for i in range(num_sweeps):
            als_start = time.time()
            print(f"Starting sweep {i}...")
            for j in range(N - 1):
                optimize_core(j)
                tt_approx.orthogonalize_push_right(j)
                tt_approx.update_internal_sampler(j, "left", True)

            for j in range(N - 1, 0, -1):
                optimize_core(j)
                tt_approx.orthogonalize_push_left(j)
                tt_approx.update_internal_sampler(j, "right", True)

            tt_approx.update_internal_sampler(0, "left", False)
            stats["als_time"] += time.time() - als_start

            if i % epoch_interval == 0 or i == num_sweeps - 1:
                fit_computation_start = time.time()
                stats["iterations"].append(i)
                fit = self.compute_exact_fit()
                stats["fits"].append(fit)
                stats["als_iter_times"].append(stats["als_time"]) 
                print(f"Fit after {i} iterations: {fit}")
                stats["fit_computation_time"] += time.time() - fit_computation_start 

        return stats 