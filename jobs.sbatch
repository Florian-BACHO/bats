#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
# SBATCH --time=8:00:00
# SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6144
# SBATCH --mail-type=END
# SBATCH --gres=gpu
source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
# module load cuda/12.0
conda activate /home/nfs/adelossantossu/.conda/envs/pytorch

srun python experiments/mnist/train_spike_count.py --gres=gpu --chdir=./out/ --mail-type=END
