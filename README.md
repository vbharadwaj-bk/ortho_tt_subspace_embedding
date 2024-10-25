# ortho_tt_subspace_embedding
This repository contains the code for
``Efficient Leverage Score Sampling for 
Tensor Train Decomposition", to appear 
at Neurips 2024.

## What Can I do with it?
You can draw samples from a 
left / right-orthonormal chain of tensor train 
cores with a ``hanging bond dimension" on
on the end of the chain. If we reshape the chain
as a matrix, each row is drawn with probability 
proportional to its squared row norm.

## Building Our Code
You will need `GCC>=8.5` and a copy of OpenBLAS. On Linux,
you can install OpenBLAS via 
`sudo apt-get install libopenblas-dev`, while you can use
`brew install openmp` on a Mac. We also strongly recommend
an install of Intel Thread Building Blocks (TBB), which you can
get via `sudo apt install libtbb-dev` or `brew install tbb`
for Linux and MacOS, respectively. Once you have installed 
these packages, you will need to locate the path to
each of them.

### Step 0: Clone the Repository
Clone the repository and `cd` into it. 
```shell
[zsh]> git clone https://github.com/vbharadwaj-bk/fast_tensor_leverage.git
[zsh]> cd fast_tensor_leverage
```

### Step 1: Install Python packages
Install Python dependencies with the following command:
```shell
[zsh]> pip install -r requirements.txt
```
We rely on the Pybind11 and cppimport packages. We
use the HDF5 format to store sparse tensors, so
you need the h5py package if you want to perform
sparse tensor decomposition. 

### Step 2: Configure the compile and runtime environments 
Within the repository, run the following command:
```shell
[zsh]> python configure.py
```
This will create two files in the repository root:
`config.json` and `env.sh`. Edit the configuration
JSON file with include / link flags for your OpenBLAS 
install. If you have TBB installed (strongly
recommended for good performance), fill in the appropriate
entries. If you do not have TBB installed (or if you
are using a compiler that automatically links the BLAS),
set those JSON entries to empty lists. 

The file `env.sh` sets up the runtime environment,
and **must be sourced every time you start a new shell** to run our code. In `env.sh`, set 
the variables CC and CXX to your 
C and C++ compilers. The C++ extension
module is compiled with these when it is imported by Python at runtime. 

### Step 3: Test the code 
You're ready to test! The C++ extension
compiles automatically the first time you run
the code, and is not compiled subsequently. Run
the following code:
```shell
[zsh]> source env.sh
[zsh]> python tt_driver.py verify_sampler 
```
If all goes well, you should see a file
`distribution_comparison.png` in the `plotting` directory that compares a set 
of samples drawn by our leverage score sampler
with the normalized true leverage score 
distribution.

To see other commands and argument options, run

```shell
[zsh]> python tt_driver.py -h 
```

## Performing Sparse Tensor Decomposition
You can find the data from our sparse tensor experiments
at <https://portal.nersc.gov/project/m1982/hdf5_sparse_tensors/>.
These are sparse tensor datasets from <frostt.io> stored in
HDF5 format (for fast reads / writes) without modification. 
Use a tool like `curl` to download a dataset, e.g. 

```shell
[zsh]> cd data 
[zsh]> curl -o uber.tns_converted.hdf5 https://portal.nersc.gov/project/m1982/hdf5_sparse_tensors/uber.tns_converted.hdf5
```
Using the `h5py` Package in Python, you can read / open these
files and examine their contents using syntax similar 
to Numpy arrays. You can decompose a sparse tensor as follows:  

```shell
[zsh]> python  tt_driver.py decompose_sparse data/uber.tns_converted.hdf5 \
            -t 40                \    
            -iter 5              \   # Number of ALS sweeps 
            -alg random          \   # Use exact instead for non-randomized LSTSQ 
            -s 65536             \   # Sampled rows for randomized algorithms
            -o outputs/test_runs \   # Output folder to place data from runs
            -e 5                 \   # Record fit (1 - rel_err) every e sweeps 
            -r 2                 \   # Repeat each experiment twice
            --overwrite          \   # Overwrite any existing files in output dir 
```
To reproduce our results, you may have to specify a preprocessing
option for the tensor values (see Larsen Kolda 2022 for an
explanation of this practice). To do this, pass the flag `--log_count` to the
command above. In our work, we used the following configurations for each tensor: 

| Tensor | Preprocessing |
|--------|---------------|
| Uber   | None          |
| Enron  | log_count     |
| NELL-2 | log_count     |

