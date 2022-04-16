# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/sample_gpt.py --input_modality audio --output_modality video --gpt_ckpt_file checkpoints/generation/A2V/AudioVideoGPT_L32_256x/epoch=54-step=599999.ckpt --output_dir output/AudioVideoGPT_L32_256x --get_audio --use_manual_annotation --num 5
