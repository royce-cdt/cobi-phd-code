#!/bin/bash
#SBATCH --account KUSCH-SL3-CPU
#SBATCH --partition cclake
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --time 00:30:00
#SBATCH -o HYP-SHORTEND08-REDO7000_model_fit.out
#SBATCH -e HYP-SHORTEND08-REDO7000_model_fit.err

# Module setup: cluster environment and recent python.
module purge
module load rhel8/default-ccl
module load python/3.11.9/gcc/nptrdpll

# Activate relevant python virtual environment.
source /rds/user/cja69/hpc-work/python/lumispy/bin/activate

# Let PyTorch use all available CPU cores for OpenMP, MKL and MKL-DNN backends
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Install required Python packages
pip install --user numpy scipy

# Run the generated Python script
python /rds/user/cja69/hpc-work/python/generated_scripts/HYP-SHORTEND08-REDO7000_fit.py
