# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/train_gpt.py --input_modality text --output_modality audio --audio_vqvae vqvae_coinrun_1024x_full_mix \
--num_workers 32 --resolution 256 --lr 0.0003 --batch_size 4 --accumulate_grad_batches 1 --max_steps 600000 --precision 32 \
--sync_batchnorm --gpus 8 --progress_bar_refresh_rate 100 --default_root_dir output/TextAudioGPT_1024x \
--use_manual_annotation --get_audio --get_text_desc --loss_audio_weight 7