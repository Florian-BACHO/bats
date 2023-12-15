#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=24:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=16384

#SBATCH --mail-type=END

#SBATCH --gres=gpu:v100

source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate /tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/envs/pytorch

srun python3 train_spike_count.py