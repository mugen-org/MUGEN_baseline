# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/sample_gpt.py --input_modality text --output_modality audio --gpt_ckpt_file checkpoints/generation/T2A/TextAudioGPT_128x/epoch=54-step=599999.ckpt --output_dir output/TextAudioGPT_128x --get_text_desc --use_manual_annotation --num 5
