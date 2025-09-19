#!/bin/bash
#
#SBATCH -p phd
#SBATCH --job-name=DiffWave
#SBATCH --output=slurm_output_%j.out
#SBATCH --gpus=3g.40g:1
#SBATCH -c 16

# SET MINICONDA_PATH HERE
MINICONDA_PATH=/data/f.caldas/miniconda3/ #EX:/home/<USER>/miniconda3/ (without any leading or trailing spaces)
export LD_LIBRARY_PATH=/data/f.caldas/miniconda3/envs/fftenv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH

if [ -z "$MINICONDA_PATH" ]
then
	self=$(basename "$0")
	echo "JOB SUBMISSION FAILED. PLEASE SET MINICONDA_PATH on $self"
else
	source "$MINICONDA_PATH"/etc/profile.d/conda.sh
	echo "Activating conda environment: fftenv"
	conda activate fftenv

	# Run the python script
	echo "Running python script: train_caldas.py"
	srun python train_caldas.py --config $1
fi
