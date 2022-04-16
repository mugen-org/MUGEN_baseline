# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/sample_gpt.py --input_modality video --output_modality audio --gpt_ckpt_file checkpoints/generation/V2A/VideoAudioGPT_L32_512x/epoch=54-step=599999.ckpt --output_dir /checkpoint/songyangzhang/mugen/VideoAudioGPT_L32_512x --get_game_frame --use_manual_annotation --num 5
