# cuQuantum with MPI (Multi GPUs)
## Interactive node
* Node Allocation
```
salloc -N 1 -C gpu -q shared_interactive --image docker:nersc/cuquantum-appliance:23.10 --module=cuda-mpich -A m1248 -t 00:30:00 --ntasks-per-node=1 -c 32 --gpus-per-task=1 --gpu-bind=none
```
* After allocation
```
conda activate gpu-aware-mpi
```
