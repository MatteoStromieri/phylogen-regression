#!/bin/bash 

#OAR -t night 
#OAR -l host=1/gpu=2,walltime=12:00:00
#OAR -p cluster=musa
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err 
#OAR -n training_job 


# display some information about attributed resources
hostname 
nvidia-smi 
 
# make use of a python torch environment
module load conda
module load cuda
module load python/3.9.13_gcc-10.4.0
conda activate newenv
python vanilla-pointnet2-cls.py