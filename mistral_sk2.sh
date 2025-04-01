#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH -A L-SLK_Dobes_25
#SBATCH --time 24:00:00    # format: HH:MM:SS
#SBATCH -N 1               # nodes
#SBATCH --job-name=mistral-sk2
#SBATCH --ntasks-per-node=1
#SBATCH --mem 64000
#SBATCH --gres=gpu:4
#SBATCH --output=mistral_sk2_%j.out    # Zachytáva stdout
#SBATCH --error=mistral_sk2_%j.err     # Zachytáva stderr

set -e

module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0
module load cuda

# python -m venv /leonardo_work/L-SLK_001_24/pbednar/venv
source /leonardo_work/L-SLK_001_24/pbednar/venv/bin/activate

# pip3 install wheel setuptools
# pip3 install torch accelerate
# pip3 install datasets
# pip3 install transformers
# pip3 install -v flash-attn --no-build-isolation
# pip3 install sentencepiece
# pip3 install bitsandbytes
# pip3 install tensorboardX

python3 /leonardo_work/L-SLK_001_24/mistral_sk2.py
