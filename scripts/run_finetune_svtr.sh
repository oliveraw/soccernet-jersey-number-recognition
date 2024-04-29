#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=soccernet-finetune-svtr
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=7:59:55
#SBATCH --account=eecs545w24_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16000m 
#SBATCH --cpus-per-gpu=3
#SBATCH --output=./logs/%x-%j.log

# The application(s) to execute along with its input arguments and options:
cd mmocr
time python tools/train.py configs/textrecog/svtr/svtr-base_20e_soccernet_gen.py --work-dir soccernet-svtr-genL-combined
cd ..
