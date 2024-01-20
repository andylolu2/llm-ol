#!/bin/bash
#!

#SBATCH --job-name cpujob
#SBATCH --account COMPUTERLAB-SL3-CPU
#SBATCH --partition icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --mail-type=NONE
#SBATCH --no-requeue
#SBATCH --output=/home/cyal4/rds/hpc-work/llm-ol/out/logs/slurm-%j.out

#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-icl              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
module load miniconda/3
module load graphviz/2.40.1

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

source $(conda info --base)/etc/profile.d/conda.sh

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

echo -e "\nExecuting command: $@\n==================\n\n"

conda run -n llm-ol $@
