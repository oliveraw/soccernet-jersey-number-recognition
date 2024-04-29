#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=soccernet-finetune-dbnetpp
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=9:59:55
#SBATCH --account=stellayu0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16000m 
#SBATCH --cpus-per-gpu=3
#SBATCH --output=./logs/%x-%j.log

# The application(s) to execute along with its input arguments and options:
cd mmocr
time python tools/train.py configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_soccernetannotated_gen.py --work-dir soccernet-dbnetpp-genL
cd ..
