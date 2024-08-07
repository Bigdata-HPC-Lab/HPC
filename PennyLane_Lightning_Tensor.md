# Installation

**Before proceeding with the installation, it is highly recommended to verify that your NVIDIA GPU supports SM 7.0 or greater and CUDA 12.0 or above. (In our case, we used the NVIDIA V100.)**

1. Create conda environment
    ```sh
    conda create -n myCondaEnv python=3.9
    ```

**Since Lightning Tensor supports cuQuantum version 24.03 and cuQuantum 24.03 supports CUDA Toolkit 12.5, it is highly recommended to proceed with CUDA Toolkit 12.5.**

2. Install CUDA and cuQuantum-python
    ```sh
    conda install -c conda-forge -c nvidia cuda-toolkit=12.5
    conda install -c conda-forge cuquantum-python=24.03
    ```

**Lightning-Qubit should be installed before Lightning-Tensor**

3. Install Lightning-Qubit
    ```sh
    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    pip install -r requirements.txt
    PL_BACKEND="lightning_qubit" pip install .
    ```

4. Install cutensornet library
    ```sh
    pip install cutensornet-cu12
    export CUQUANTUM_SDK=$(python -c "import site; print(f'{site.getsitepackages()[0]}/cuquantum/lib')")
    ```

5. Install Lightning-Tensor
    ```sh
    PL_BACKEND="lightning_tensor" pip install -e .
    ```
