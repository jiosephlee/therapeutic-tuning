#!/usr/bin/bash
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=160G

nvidia-smi

module load cuda/12.4
module load python/3.11

# activate the venv
source ${HOME}/Venvs/torch/bin/activate

./fine-tuning-or-retrieval/scripts/MEDEX/fine_tuning_on_medex_v1.py
