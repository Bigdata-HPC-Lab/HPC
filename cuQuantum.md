


#module load and activate conda
module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python 
conda create -n gpu-aware-mpi python -y
conda activate gpu-aware-mpi

#install mpi4py 
MPICC="cc -shared" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

pip install cupy-cuda11X 
pip install -v --no-cache-dir cuquantum

#download and install cuquantum benchmark
wget https://github.com/NVIDIA/cuQuantum/archive/refs/tags/v23.10.0.tar.gz
tar -xf v23.10.0.tar.gz
cd cuQuantum-23.10.0/benchmarks/
pip install .
