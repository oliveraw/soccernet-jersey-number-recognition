#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=soccernet-kmeans-colors
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=7:00:00
#SBATCH --account=eecs545w24_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16000m 
#SBATCH --cpus-per-gpu=3
#SBATCH --output=./logs/%x-%j.log

# The application(s) to execute along with its input arguments and options:
time python run_kmeans_on_bboxes.py
