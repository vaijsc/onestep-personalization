#!/bin/bash
#SBATCH --partition=research
#SBATCH --output=/lustre/scratch/client/movian/research/users/tungnt132/SwiftPersonalize/sbatch/out_debug/out2.out
#SBATCH --job-name=bop_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Accept the script name as an argument
SCRIPT_NAME=$1

srun /bin/bash -c \
    "
    export HTTP_PROXY=http://proxytc.vingroup.net:9090/
    export HTTPS_PROXY=http://proxytc.vingroup.net:9090/
    export http_proxy=http://proxytc.vingroup.net:9090/
    export https_proxy=http://proxytc.vingroup.net:9090/

    source /sw/software/miniconda3/bin/activate
    conda activate /lustre/scratch/client/vinai/users/tungnt132/conda_env/swift_per
    cd /lustre/scratch/client/movian/research/users/tungnt132/SwiftPersonalize
    sh scripts/$SCRIPT_NAME.sh
    "