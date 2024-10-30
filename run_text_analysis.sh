#!/bin/bash
#SBATCH --job-name text-analysis.job
#SBATCH --output text-analysis.out
#SBATCH --time 1:00:00
#SBATCH --mem 16G
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1

module purge
module load anaconda/3/2023.03
module load cuda/12.1
source /u/alehe/anaconda3/etc/profile.d/conda.sh
conda activate gen-culture-2

cd /raven/u/alehe/Projects/track-creativity/gutenberg

# Run the text analysis script
python text_analysis.py "Wilde, Oscar"
