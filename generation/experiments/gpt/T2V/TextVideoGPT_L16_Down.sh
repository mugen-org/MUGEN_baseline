# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/train_gpt.py --input_modality text --output_modality video \
--video_vqvae checkpoints/generation/video_vqvae/L16/epoch=54-step=599999.ckpt \
--get_game_frame --get_text_desc --use_downsampled_trainset \
--num_workers 32 --resolution 256 --sequence_length 16 --sample_every_n_frames 6 --lr 0.0003 --batch_size 4 \
--accumulate_grad_batches 1 --max_steps 600000 --precision 32 --sync_batchnorm --gpus 8 --progress_bar_refresh_rate 100 \
--default_root_dir output/TextVideoGPT_L16_Down --use_manual_annotation --loss_video_weight 7
