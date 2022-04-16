# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/eval_by_cap.py --input_modality audio --output_modality text --get_text_desc --use_manual_annotation --output_dir output/TextAudioGPT_256x --cap_ckpt_file checkpoints/generation/A2T/AudioTextGPT_128x/epoch=54-step=599999.ckpt
