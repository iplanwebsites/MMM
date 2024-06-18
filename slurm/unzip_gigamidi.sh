#!/bin/bash

#SBATCH --job-name=unzip-data
#SBATCH --time=20:00:00
#SBATCH --account=def-pasquier
#SBATCH --mail-user=raa60@sfu.ca # Default mail
#SBATCH --nodes=1            # total nb of nodes
#SBATCH --ntasks-per-node=1  # nb of tasks per nod
#SBATCH --cpus-per-task=64
#SBATCH --mem=249G

free -h


cd $SCRATCH/data/Cleaned_Ver_EP_Class-GigaMIDI
zip -d Cleaned_Ver_GigaMIDI.zip __MACOSX/\*
zip -d Cleaned_Ver_GigaMIDI.zip \*/.DS_Store
unzip Cleaned_Ver_GigaMIDI.zip
