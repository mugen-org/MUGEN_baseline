#!/bin/bash
#SBATCH --job-name=audio_text_retrieval
#SBATCH --output=/checkpoint/%u/mugen/%x/%j.out
#SBATCH --error=/checkpoint/%u/mugen/%x/%j.err
#SBATCH --partition=multigen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=480gb
#SBATCH --time=14-00:00:00 \
#SBATCH --constraint=volta32gb
#SBATCH --mail-user=songyangzhang@fb.com
#SBATCH --mail-type=end

python main.py --model_name audio_text_retrieval \
  --batch_size 16 --text_enc --audio_enc --trainable --pretrained \
  --default_root_dir output/audio_text_retrieval \
  --get_audio --get_text_desc --use_manual_annotation