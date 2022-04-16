# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/sample_gpt.py --input_modality audio --output_modality text --gpt_ckpt_file checkpoints/generation/A2T/AudioTextGPT_128x/epoch=54-step=599999.ckpt --output_dir output/AudioTextGPT_128x --get_audio --use_manual_annotation  --top_k 1 --top_p 0.5
