#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
sbatch --time=8:00:00
sbatch --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6144
sbatch --mail-type=END
sbatch --gres=gpu

module use /opt/insy/modulefiles
module load miniconda/3.9
module load cuda/12.0

srun python experiments/mnist/train_spike_count.py
