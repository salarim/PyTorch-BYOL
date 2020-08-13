#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_gpu
#SBATCH --job-name=BYOL-STL10
#SBATCH --output=%x-%j.out
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --mem=10000M
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=16

cd $SLURM_TMPDIR
cp -r ~/scratch/PyTorch-BYOL .
cd PyTorch-BYOL
rm -r runs

module load python/3.7 cuda/10.0
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python main.py

cp -r runs ~/scratch/PyTorch-BYOL

