# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/train_video_vqvae.py --model VideoVQVAE \
--embedding_dim 256 --n_codes 2048 --n_hiddens 240 --n_res_layers 4 --downsample 4 32 32 --kernel_size 3 \
--num_workers 32 --resolution 256 --sequence_length 32 --sample_every_n_frames 3 --lr 0.0003 --batch_size 4 \
--max_steps 600000 --gradient_clip_val 1 --gpus 8 --progress_bar_refresh_rate 200 --sync_batchnorm \
--default_root_dir output/VideoVQVAE_L32 --get_game_frame