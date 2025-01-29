# bvmodelgen2

For using fiber and UVC generator do:
1. Create a conda environment: `conda create --name bvgen2`
2. Activate the environment: `conda activate bvgen2`
3. Install FEniCSx: `conda install -c conda-forge fenics-dolfinx mpich h5py cffi python`
4. Install bvmodelgen2, do `python -m pip install -e .` inside the repository folder (make sure the environment is activated).

If you do not need to use the fiber and UVC generator:
1. Create a conda environment: `conda create --name bvgen2 python`
2. Activate the environment: `conda activate bvgen2`
4. Install bvmodelgen2, do `python -m pip install -e .` inside the repository folder (make sure the environment is activated).