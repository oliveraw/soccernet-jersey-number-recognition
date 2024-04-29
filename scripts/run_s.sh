#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=soccernet-gen-finetuned-dbnetpp30e-svtr20e
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=20:00:00
#SBATCH --account=stellayu0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16000m 
#SBATCH --cpus-per-gpu=3
#SBATCH --output=./logs/%x-%j.log

# The application(s) to execute along with its input arguments and options:
time python run_jocelyn.py --det_threshold 0.6 --rec_threshold 0.0
