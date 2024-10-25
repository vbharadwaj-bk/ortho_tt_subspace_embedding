import numpy as np
import numpy.linalg as la

import logging, sys, time, argparse, datetime, os, json

import cppimport
import cppimport.import_hook

from tensors.tensor_train import *
from tensors.sparse_tensor import *
from algorithms.tt_als import *

#from tensor_io.torch_tensor_loader import get_torch_tensor

def verify_sampler(args, I=20, R=4, N=3, J=10000, seed=20, test_direction="left"): 
    '''
    Test that our algorithm can match the true leverage
    score distribution from the left and right subchains
    '''
    import matplotlib.pyplot as plt

    dims = [I] * N 
    ranks = [R] * (N-1) 
    tt = TensorTrain(dims, ranks, seed)

    if test_direction == "left":
        tt.place_into_canonical_form(N-1)
        tt.build_fast_sampler(N-1, J)
        samples = tt.leverage_sample(j=N-1, J=J, direction="left")
        linear_idxs = np.array(tt.linearize_idxs_left(samples), dtype=np.int64)
        left_chain = tt.left_chain_matricize(N-1)
        normsq_rows_left = la.norm(left_chain, axis=1) ** 2
        normsq_rows_normalized_left = normsq_rows_left / np.sum(normsq_rows_left)
        true_dist = normsq_rows_normalized_left
    else:
        tt.place_into_canonical_form(0)
        tt.build_fast_sampler(0, J)
        samples = tt.leverage_sample(j=0, J=J, direction="right")
        linear_idxs = np.array(tt.linearize_idxs_right(samples), dtype=np.int64)
        right_chain = tt.right_chain_matricize(0)
        normsq_rows_right = la.norm(right_chain, axis=1) ** 2 
        normsq_rows_normalized_right = normsq_rows_right / np.sum(normsq_rows_right)
        true_dist = normsq_rows_normalized_right

    fig, ax = plt.subplots()
    ax.plot(true_dist, label="True leverage distribution")
    bins = np.array(np.bincount(linear_idxs, minlength=len(true_dist))) / J

    ax.plot(bins, label="Our sampler")
    ax.set_xlabel("Row Index")
    ax.set_ylabel("Probability Density")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"{J} samples, our method vs. true TT-chain leverage distribution")
    fig.savefig('plotting/distribution_comparison.png')

def test_sparse_tensor_decomposition(params):
    tensor_name = params.tensor.split('/')[-1]
    preprocessing = None

    if params.log_count:
        preprocessing = "log_count"

    filename_prefix = '_'.join([tensor_name, str(params.trank), 
                                    str(params.iter), params.algorithm, str(params.samples), 
                                    str(params.epoch_iter)])

    files = os.listdir(args.output_folder)
    filtered_files = [f for f in files if filename_prefix in f]

    trial_nums = []
    for f in filtered_files:
        with open(os.path.join(args.output_folder, f), 'r') as f_handle:
            exp = json.load(f_handle)
            trial_nums.append(exp["trial_num"])

    remaining_trials = list(range(args.repetitions))  
    if not params.overwrite:
        remaining_trials = [i for i in range(args.repetitions) if i not in trial_nums]

    if len(remaining_trials) > 0:
        trial_num = remaining_trials[0] 
        output_filename = f'{filename_prefix}_{trial_num}.out'

    if len(remaining_trials) == 0:
        print("No trials left to perform!")
        exit(0)

    for trial_num in remaining_trials:
        output_filename = f'{filename_prefix}_{trial_num}.out'
        print(f"Starting trial {output_filename}")

        ground_truth = PySparseTensor(params.tensor, lookup="sort", preprocessing=preprocessing)

        ranks = [params.trank] * (ground_truth.N - 1)

        print("Loaded dataset...")
        tt_approx = TensorTrain(ground_truth.shape, ranks)

        tt_approx.place_into_canonical_form(0)
        tt_approx.build_fast_sampler(0, J=params.samples)
        tt_als = TensorTrainALS(ground_truth, tt_approx)
        optimizer_stats = None

        initial_fit = tt_als.compute_exact_fit()

        if params.algorithm == "exact":
            optimizer_stats = tt_als.execute_exact_als_sweeps_sparse(num_sweeps=params.iter, J=params.samples, epoch_interval=params.epoch_iter)
        elif params.algorithm == "random":
            optimizer_stats = tt_als.execute_randomized_als_sweeps(num_sweeps=params.iter, J=params.samples, epoch_interval=params.epoch_iter)

        final_fit = tt_als.compute_exact_fit()

        now = datetime.datetime.now()
        output_dict = {
            'time': now.strftime('%m/%d/%Y, %H:%M:%S'), 
            'input': params.tensor,
            'preprocessing': preprocessing, 
            'target_rank': params.trank,
            'iterations': params.iter,
            'algorithm': params.algorithm,
            'sample_count': params.samples,
            'accuracy_epoch_length': params.epoch_iter,
            'trial_count': params.repetitions,
            'trial_num': trial_num,
            'initial_fit': initial_fit,
            'final_fit': final_fit,
            'thread_count': os.environ.get('OMP_NUM_THREADS'),
            'stats': optimizer_stats
        }

        print(json.dumps(output_dict, indent=4))
        print(f"Final Fit: {final_fit}")

        if output_filename is not None:
            with open(os.path.join(args.output_folder, output_filename), 'w') as f:
                f.write(json.dumps(output_dict, indent=4)) 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='subcommand help')
    parser_verify = subparsers.add_parser('verify_sampler', help='Verifies sampler correctness by plotting true leverge samples vs. our samples in plotting directory.')
    parser_verify.set_defaults(func=verify_sampler)

    parser_decompose = subparsers.add_parser('decompose_sparse', help='Decomposes a sparse tensor.')
    parser_decompose.add_argument('tensor', type=str, help='Path to tensor to decompose')
    parser_decompose.add_argument("-t", "--trank", help="Rank of the target decomposition", required=True, type=int)
    parser_decompose.add_argument("-iter", help="Number of ALS iterations", required=True, type=int)
    parser_decompose.add_argument('-alg','--algorithm', type=str, help='Algorithm to perform decomposition', choices=['exact', 'random'], required=True)
    parser_decompose.add_argument("-s", "--samples", help="Number of samples taken from the KRP", required=False, type=int)
    parser_decompose.add_argument("-o", "--output_folder", help="Folder name to print statistics", required=False)
    parser_decompose.add_argument("-e", "--epoch_iter", help="Number of iterations per accuracy evaluation epoch", required=False, type=int, default=5)
    parser_decompose.add_argument("-r", "--repetitions", help="Number of repetitions for multiple trials", required=False, type=int, default=1)
    parser_decompose.add_argument("--log_count", help="Take the log of the values", action="store_true")
    parser_decompose.add_argument("--overwrite", help="Overwrite any existing stat files in output directory", action="store_true")

    parser_decompose.set_defaults(func=test_sparse_tensor_decomposition)

    args = parser.parse_args()
    args.func(args) 