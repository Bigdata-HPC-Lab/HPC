# cuQuantum with MPI (Multi GPUs)
## Configureation
* Setup (Module load and Activate Conda)
```
module  load  PrgEnv-nvidia  cray-mpich  cudatoolkit/12.2  craype-accel-nvidia80  python/3.9
conda  create  -n  gpu-aware-mpi  python=3.9  -y
conda  activate  gpu-aware-mpi
```
* Install mpi4py, cupy, cuStateVec, and cuTensorNet.
```
MPICC="cc -shared" CC=nvc CFLAGS="-noswitcherror" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
pip  install  cupy-cuda12X
conda install -c conda-forge custatevec
conda install -c conda-forge cutensornet "mpich=*=external_*"
```
* Download and install cuQuantum benchmark.
```
wget  https://github.com/NVIDIA/cuQuantum/archive/refs/tags/v23.10.0.tar.gz
tar  -xf  v23.10.0.tar.gz
cd cuQuantum-23.10.0/python
pip install -e .
cd cuQuantum-23.10.0/benchmark
pip install -e .
```
* Execute cuQuantum benckmark.
```
cuquantum-benchmarks circuit --frontend cirq --backend cutn --benchmark qft --nqubits 8 --ngpus 1
```

## Enabled GPU-aware cuQuantum benchmark
* The following content needs to be applied.
1) To set up a Multi-GPU environment using cuTensorNet with MPI, cray-mpich must be loaded, and in Perlmutter, cray-mpich is already configured.
2) However, even after loading the module, a GTL error occurs, preventing the MPI environment from being properly configured. Therefore, it is necessary to specify the path in the LD_LIBRARY_PATH explicitly.
3) Since cuTensorNet uses the MPI library from the cray-mpich we are using via CUTENSORNET_COMM_LIB, this environment variable must also be specified.
4) Lastly, the MPI library that supports GTL networking needs to be preloaded using LD_PRELOAD to ensure proper allocation.

* Example(Sbatch File - Sbatch_mpi.sh) 
```
#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q debug
#SBATCH -J kcj
#SBATCH --mail-user=changjong5238@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00
#SBATCH -A m1248


# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread



# Configure (Module load)
module  load  PrgEnv-nvidia  cudatoolkit/12.2  craype-accel-nvidia80  python/3.9
module load cray-mpich/8.1.28

# Configure (To utilize GTL Library of MPI)
module unload craype-accel-nvidia80
module load craype-accel-nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
#export CRAY_ACCEL_TARGET=nvidia80

# Activate conda 
conda activate kcj_slice

# Configure PATH related to Conda 
export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2
export CONDA_PREFIX=/global/homes/s/sgkim/.conda/envs/kcj_slice

# LD_PRELOAD and LD_LIBRARY_PATH
export LD_PRELOAD=/global/homes/s/sgkim/.conda/envs/kcj_original/lib/libmpi_gtl_cuda.so
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH:/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/lib:/opt/cray/pe/mpich/8.1.28/gtl/lib

# CUTENSORNET_COMM_LIB Home Path Explicitly Set
export CUTENSORNET_COMM_LIB=/global/homes/s/sgkim/.conda/envs/kcj_slice/lib/libcutensornet_distributed_interface_mpi.so

# GPU Allocation 
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=$(($SLURM_LOCALID % 4))

#KCJ complete (sample code)
#srun -n 4 -c 32 --cpu_bind=cores -G 4 --gpu-bind=none python example22_mpi_auto.py > output.log 2>&1

#KCJ complete (cuquantum)
srun -n 4 --gpus-per-task=1 cuquantum-benchmarks circuit --frontend cirq --backend cutn --benchmark qft --nqubits 32 --ngpus 1 > output.log 2>&1

cat output.log

``` 
* To set up a symbolic link for the GTL library in a Conda environment (1)
```
cd /opt/cray/pe/mpich/8.1.28/gtl/lib

# Set up a symbolic link in your Conda_HOME/lib.
ln -sf /opt/cray/pe/mpich/8.1.28/gtl/lib/libmpi_gtl_cuda.so ~/.conda/envs/Kcj_conda/lib/libmpi_gtl_cuda.so

# Use ldd to check if libmpi_gti_cuda.so is linked with mpi4py.
cd ~/.conda/envs/kcj_slice/lib/python3.9/site-packages/mpi4py
ldd MPI.cpython-39-x86_64-linux-gnu.so
```
![image](https://github.com/user-attachments/assets/85e6b852-36d0-4993-b491-d49b2778043b)


* To set up a symbolic link for the GTL library in a Conda environment (2) - Example: Sbatch_mpi.sh
  - MPICH_GPU_SUPPORT_ENABLED=1
  - For more details, refer to the NERSC Documentation: https://docs.nersc.gov/development/programming-models/mpi/cray-mpich/)
  - At run time MPICH_GPU_SUPPORT_ENABLED=1 must be set. If it is not set there will be Errors similar to
```
MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested, but GTL library is not linked

```

```
# Configure (To utilize GTL Library of MPI)
module unload craype-accel-nvidia80
module load craype-accel-nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
#export CRAY_ACCEL_TARGET=nvidia80

# LD_PRELOAD and LD_LIBRARY_PATH
export LD_PRELOAD=/global/homes/s/sgkim/.conda/envs/kcj_original/lib/libmpi_gtl_cuda.so
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH:/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/lib:/opt/cray/pe/mpich/8.1.28/gtl/lib
```


* Use a symbolic link for the MPI library with CUTENSORNET_MPI_COMM. (1)

```
cd /opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/lib-abi-mpich
ln -sf /opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/lib-abi-mpich/libmpi.so.12 ~/.conda/envs/Kcj_conda/lib/libmpi.so.12

cd ~/.conda/envs/Kcj_conda/lib/
ldd libcutensornet_distributed_interface_mpi.so
``` 
![image](https://github.com/user-attachments/assets/170a42a2-d8bb-475e-9877-0dc8feadcd51)

* If MPI is not properly linked, it will be as follows.
![image](https://github.com/user-attachments/assets/fdc41f8e-e6c6-4a90-9ef5-a4fc43dea444)


* Use a symbolic link for the MPI library with CUTENSORNET_MPI_COMM. (2) - Example: Sbatch_mpi.sh
```
# CUTENSORNET_COMM_LIB Home Path Explicitly Set
export CUTENSORNET_COMM_LIB=/global/homes/s/sgkim/.conda/envs/kcj_slice/lib/libcutensornet_distributed_interface_mpi.so
```


* GPU Allocation - Example: Sbatch_mpi.sh
  - You can use SLURM_LOCAL_ID, but since it does not provide proper allocation here, it is set up as follows.
```
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Using SLURM_LOCAL_ID
#export CUDA_VISIBLE_DEVICES=$(($SLURM_LOCALID % 4))
#export CUDA_VISIBLE_DEVICES=$(echo $SLURM_LOCALID | awk '{printf "%s", $1%4}')
```
* Execute cuQuantum benchmark - Example: Sbatch_mpi.sh
```
srun -n 4 --gpus-per-task=1 cuquantum-benchmarks circuit --frontend cirq --backend cutn --benchmark qft --nqubits 32 --ngpu    s 1 
```


## How to run MPI with the example code.
* Execute example code (example22_mpi_auto.py) (1)
  - Path(ex): /pscratch/sd/s/sgkim/kcj_cuquantum_only_original/cuQuantum-24.03.0/python/samples/cutensornet/coarse
  - Below is a part of example22_mpi_auto.py. The following section should be commented out and the code modified as shown.
```

 # Broadcast the operand data. Throughout this sample we take advantage of the upper-case mpi4py APIs
 # that support communicating CPU & GPU buffers (without staging) to reduce serialization overhead for array-like objects. This capability requires mpi4py v3.10+.
 #for operand in operands:
 #   comm.Bcast(operand, root)

----- Modification
# For each operand, we:
# 1. Create a buffer `buf`. On the root process, `buf` is filled with the data from `operand`. On other processes, `buf` is an empty `numpy` array with the same shape and dtype as `operand`.
# 2. Broadcast the buffer `buf` to all processes using `comm.Bcast`.
# 3. If the process is not the root, convert `buf` from a `numpy` array to a GPU array and store it back in `operands[i]`.
# This approach ensures compatibility across different `mpi4py` versions and handles data transfer without relying on specific upper-case MPI APIs.

import numpy as np 

for i, operand in enumerate(operands):
    buf = operand.get() if rank == root else np.empty(operand.shape, dtype=operand.dtype)
    comm.Bcast(buf, root)
    if rank != root:
        operands[i] = cp.asarray(buf)


```
  
* Execute example code (example22_mpi_auto.py) (2) - Example: Sbatch_mpi.sh
```
srun -n 4 --gpus-per-task=1 python example22_mpi_auto.py 
```
  






