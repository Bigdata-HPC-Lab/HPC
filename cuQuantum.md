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

# Testing in other nodes
login node

    shifter --image  docker:nvcr.io/nvidia/cuquantum-appliance:23.10 cuquantum-benchmarks  circuit  --frontend  cirq  --backend  cutn  --benchmark  qft  --nqubits  1  --ngpus  1

interactive node

    salloc -N 1 -C gpu -q shared_interactive --image docker:nvcr.io/nvidia/cuquantum-appliance:23.10 -A m1248 -t 04:00:00 --ntasks-per-node=1 -c 32 --gpus-per-task=1 --gpu-bind=none
    shifter activate gpu-aware-mpi
    shifter --module=cuda-mpich /global/homes/s/sgkim/.local/perlmutter/python-3.11/bin/cuquantum-benchmarks circuit --frontend  cirq  --backend  cutn  --benchmark  qft  --nqubits  1  --ngpus  1

sbatch

    #!/bin/bash
    #SBATCH --image=docker:nvcr.io/nvidia/cuquantum-appliance:23.10
    #SBATCH -N 1
    #SBATCH -C gpu
    #SBATCH -q debug
    #SBATCH -J kcj
    #SBATCH --mail-type=ALL
    #SBATCH -t 00:03:00
    #SBATCH -A m1248
    #SBATCH --ntasks-per-node=4
    #SBATCH -c 32
    #SBATCH --gpus-per-task=1
    #SBATCH --gpu-bind=none
    
    export SLURM_CPU_BIND="cores"
    module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python
    shifter activate gpu-aware-mpi
    srun shifter --module=cuda-mpich /global/homes/s/sgkim/.local/perlmutter/python-3.11/bin/cuquantum-benchmarks circuit --frontend  cirq  --backend  cutn  --benchmark  qft  --nqubits  1  --ngpus  4
