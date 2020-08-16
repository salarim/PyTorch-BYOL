#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_gpu
#SBATCH --job-name=BYOL-cifar10-m0.99-lr0.1-mlp-proj256-hidden4096-e100-wd1e-4-myresnet
#SBATCH --output=%x-%j.out
#SBATCH --ntasks=1
#SBATCH --time=2:59:00
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

python eval.py --model-path runs/*/checkpoints/model.pth

cp -r runs ~/scratch/PyTorch-BYOL

