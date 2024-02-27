#!/bin/bash
#!

#SBATCH --job-name cpujob
#SBATCH --account COMPUTERLAB-SL2-CPU
#SBATCH --partition icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
#SBATCH --no-requeue
#SBATCH --output=/home/cyal4/tmp/slurm/interactive-%j.out

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

sleep infinity
