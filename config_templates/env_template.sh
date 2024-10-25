export CC=gcc
export CXX=g++

export MACOSX_DEPLOYMENT_TARGET=11.3

# Set based on your system configuration 
export OMP_NUM_THREADS=16      
export OMP_MAX_ACTIVE_LEVELS=2

# Avoids oversubscription of resources due to nested
# parallelism. Critical for good performance!
export OPENBLAS_NUM_THREADS=1