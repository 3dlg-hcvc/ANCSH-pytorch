#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:p100:1         # Number of GPUs (per node)
#SBATCH --mem=32000               # memory (per node)
#SBATCH --time=2-23:00            # time (DD-HH:MM)
#SBATCH --cpus-per-task=6         # Number of CPUs (per task)
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=shawn_jiang@sfu.ca
#SBATCH --output=/home/hanxiao/scratch/proj-motionnet/ancsh_output/npcs/%x_%j.out
#SBATCH --job-name=npcs
echo 'Start'

echo 'ENV Start'

module load StdEnv/2020  intel/2020.1.217
module load python/3.7
module load cuda/11.0
module load cudnn/8.0.3

source /home/hanxiao/scratch/proj-motionnet/ancsh_env/bin/activate
export PROJ_DIR=/home/hanxiao/projects/rrg-msavva/hanxiao/proj-motionnet/ANCSH-pytorch

echo 'Job Start'
python $PROJ_DIR/train.py network=npcs paths=cc_npcs train.input_data=/home/hanxiao/projects/rrg-msavva/hanxiao/proj-motionnet/Dataset/eyeglasses_ancsh/train.h5 test.input_data=/home/hanxiao/projects/rrg-msavva/hanxiao/proj-motionnet/Dataset/eyeglasses_ancsh/test.h5