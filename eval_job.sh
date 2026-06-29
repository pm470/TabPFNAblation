#!/bin/bash
#SBATCH --job-name=nanotabpfn_train
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pp216@email.uni-freiburg.de

module load devel/python/3.11
module load devel/cuda/12.8
source ~/tabpfn_env/bin/activate
cd ~/TabPFNAblation/
python eval_openml.py
