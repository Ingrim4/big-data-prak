#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --partition=clara
#SBATCH --time=00:20:00
#SBATCH --job-name=roof_train_test
#SBATCH -o logs/%x-%j/out.log
#SBATCH -e logs/%x-%j/err.log

module load CUDA/11.7.0

source env/bin/activate
srun python src/train.py --num-gpus 2