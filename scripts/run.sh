#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=soccernet
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8000m 
#SBATCH --time=7:59:55
#SBATCH --account=eecs545w24_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=./logs/%x-%j.log

# The application(s) to execute along with its input arguments and options:
time python run.py
