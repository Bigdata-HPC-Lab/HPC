
# Installation

1. module load and activate conda
```
module  load  PrgEnv-gnu  cray-mpich  cudatoolkit  craype-accel-nvidia80  python
conda  create  -n  gpu-aware-mpi  python  -y
conda  activate  gpu-aware-mpi
```
* To reset conda environment and start again
```
conda deactivate
conda  env  remove  -n  gpu-aware-mpi
```
2. install mpi4py and cupy
```
MPICC="cc -shared"  pip  install  --force  --no-cache-dir  --no-binary=mpi4py  mpi4py
pip  install  cupy-cuda11X
```
3. Install cuquantum
```
conda install -c conda-forge cutensornet mpich
```
4. Download and install cuquantum benchmark
```
wget  https://github.com/NVIDIA/cuQuantum/archive/refs/tags/v23.10.0.tar.gz
tar  -xf  v23.10.0.tar.gz
cd  cuQuantum-23.10.0/benchmarks/

#need to deactivate conda first
conda deactivate
pip  install  .[all]
```
4. Execute benchmark 
```
/global/homes/s/sgkim/.local/perlmutter/python-3.11/bin/cuquantum-benchmarks circuit  --frontend  cirq  --backend  cutn  --benchmark  qft  --nqubits  1  --ngpus  1
```
# Shifter Images
* Official image from nvidia
```
--image=docker:nvcr.io/nvidia/cuquantum-appliance:23.10
```
* Image from Nersc. cuquantum + explict installation of mpi4py for distributed in perlmutter (Using this one)
```
--image=docker:registry.nersc.gov/library/nersc/cuquantum:cuda-11.7
```
ETC.
'''
conda install -c conda-forge pycocotools

Error: ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /global/homes/s/sgkim/.local/perlmutter/python-3.11/lib/python3.11/site-packages/cuquantum/custatevec/custatevec.cpython-311-x86_64-linux-gnu.so)

'''

# Testing in other nodes
## login node (with swifter)
```
shifter --image docker:registry.nersc.gov/library/nersc/cuquantum:cuda-11.7 /global/homes/s/sgkim/.local/perlmutter/python-3.11/bin/cuquantum-benchmarks circuit  --frontend  cirq  --backend  cutn  --benchmark  qft  --nqubits  1  --ngpus  1
```
## interactive node
* Node Allocation
```
salloc -N 1 -C gpu -q shared_interactive --image docker:nersc/cuquantum-appliance:23.10 --module=cuda-mpich -A m1248 -t 01:00:00 --ntasks-per-node=1 -c 32 --gpus-per-task=1 --gpu-bind=none
```
* (Serial) Run inside interactive node 
```
conda activate gpu-aware-mpi
shifter activate gpu-aware-mpi
shifter --module=cuda-mpich /global/homes/s/sgkim/.local/perlmutter/python-3.11/bin/cuquantum-benchmarks circuit --frontend  cirq  --backend  cutn  --benchmark  qft  --nqubits  1  --ngpus  1
```
* (Parallel) Run with srun (Does NOT work)
```
export SLURM_CPU_BIND="cores"
conda activate gpu-aware-mpi
shifter activate gpu-aware-mpi
srun shifter --module=cuda-mpich /global/homes/s/sgkim/.local/perlmutter/python-3.11/bin/cuquantum-benchmarks circuit --frontend  cirq  --backend  cutn  --benchmark  qft  --nqubits  1  --ngpus  1
```
* Error
```
  File "/global/homes/s/sgkim/.local/perlmutter/python-3.11/lib/python3.11/site-packages/cuquantum_benchmarks/_utils.py", line 168, in is_running_mpi
    raise RuntimeError(
RuntimeError: it seems you are running mpiexec/mpirun but mpi4py cannot be imported, maybe you forgot to install it?
srun: error: nid200296: task 0: Exited with exit code 1
```

## salloc and srun (Does NOT work, same problem as interactive node)
```
salloc -N 1 -C gpu -q shared --image docker:registry.nersc.gov/library/nersc/cuquantum:cuda-11.7 --module=cuda-mpich -A m1248 -t 00:10:00 --ntasks-per-node=1 -c 32 --gpus-per-task=1 --gpu-bind=none
```
```
shifter --module=cuda-mpich activate gpu-aware-mpi
srun -N 1 -n 4 shifter --module=cuda-mpich activate gpu-aware-mpi && /global/homes/s/sgkim/.local/perlmutter/python-3.11/bin/cuquantum-benchmarks circuit --frontend  cirq  --backend  cutn  --benchmark  qft  --nqubits  1  --ngpus  1
```

## sbatch mode (NOT WORKING)
```
sbatch run.batch
```
* run.batch example
```
#!/bin/bash
#SBATCH --image=docker:registry.nersc.gov/library/nersc/cuquantum:cuda-11.7
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -J kcj
#SBATCH -t 00:03:00
#SBATCH -A m1248
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"
module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python
srun shifter --module=cuda-mpich /global/homes/s/sgkim/.local/perlmutter/python-3.11/bin/cuquantum-benchmarks circuit --frontend  cirq  --backend  cutn  --benchmark  qft  --nqubits  1  --ngpus 1
```

## Bugfix
```
  File "/global/homes/s/sgkim/.local/perlmutter/python-3.11/lib/python3.11/site-packages/cuquantum_benchmarks/backends/backend_cutn.py", line 44, in __init__
    cutn.distributed_reset_configuration(
  File "cuquantum/cutensornet/cutensornet.pyx", line 2699, in cuquantum.cutensornet.cutensornet.distributed_reset_configuration
  File "cuquantum/cutensornet/cutensornet.pyx", line 2721, in cuquantum.cutensornet.cutensornet.distributed_reset_configuration
  File "cuquantum/cutensornet/cutensornet.pyx", line 326, in cuquantum.cutensornet.cutensornet.check_status
cuquantum.cutensornet.cutensornet.cuTensorNetError: CUTENSORNET_STATUS_DISTRIBUTED_FAILURE
```

refer to this [cuQuantum discussion](https://github.com/NVIDIA/cuQuantum/discussions/30) <- EXTERMELY IMPORTANT, also has information about CPU - GPU blinding using rack and GPU device ID /// also look at [summary post](https://github.com/NVIDIA/cuQuantum/issues/31)

check libmpi status and fix libmpi.so.12 not found error
```
ldd ~/.conda/envs/gpu-aware-mpi/lib/libcutensornet_distributed_interface_mpi.so
    libmpi.so.12 => not found 
```

```
/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib-abi-mpich/
/global/homes/s/sgkim/mpich-4.0.2/lib/.libs
--env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi/lib
shifter MPICC="cc -shared" pip install cuquantum
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib-abi-mpich/:/usr/local/openmpi/lib
module  load  PrgEnv-gnu  cray-mpich  cudatoolkit  craype-accel-nvidia80  python
conda  activate  gpu-aware-mpi
conda uninstall cutensornet
conda uninstall openmpi
conda install -c conda-forge "mpich=8.1.25=external_*"
conda install -c conda-forge cutensornet "mpich=8.1.25=external_*"
```

# cuQuantum Benchmarks exeuctions (change the last srun command from the .batch file)
ToDo
1. increase nqubits
2. increase ngpus to 4 (4 gpus per node)
3. increase -N to 4 with ngpus to 16

multiGPU simulation (Does NOT work, need to explore "mpiexec" and backend from cuquantum benchmark)
```
srun shifter --module=cuda-mpich \
activate gpu-aware-mpi \
&& mpiexec -n 4 /global/homes/s/sgkim/.local/perlmutter/python-3.11/bin/cuquantum-benchmarks circuit --frontend qiskit --backend cusvaer --benchmark quantum_volume --nqubits 8 --ngpus 2 --cusvaer-global-index-bits 1,1 --cusvaer-p2p-device-bits 1
```
* mpiexec -n #  <- this should match SBATCH -N # count from .batch file


