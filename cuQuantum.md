# module load and activate conda

    module  load  PrgEnv-gnu  cray-mpich  cudatoolkit  craype-accel-nvidia80  python
    conda  create  -n  gpu-aware-mpi  python  -y
    conda  activate  gpu-aware-mpi

conda  env  remove  -n  gpu-aware-mpi

# install mpi4py and cupy

    MPICC="cc -shared"  pip  install  --force  --no-cache-dir  --no-binary=mpi4py  mpi4py
    pip  install  cupy-cuda11X

# Download and install cuquantum benchmark

    wget  https://github.com/NVIDIA/cuQuantum/archive/refs/tags/v23.10.0.tar.gz
    tar  -xf  v23.10.0.tar.gz
    cd  cuQuantum-23.10.0/benchmarks/
    
    #need to deactivate conda first
    conda deactivate
    pip  install  .[all]

  
# Execute benchmark 

    cuquantum-benchmarks  circuit  --frontend  cirq  --backend  cutn  --benchmark  qft  --nqubits  1  --ngpus  1

# Testing in login node

    shifter --image  docker:nvcr.io/nvidia/cuquantum-appliance:23.10 cuquantum-benchmarks  circuit  --frontend  cirq  --backend  cutn  --benchmark  qft  --nqubits  1  --ngpus  1
