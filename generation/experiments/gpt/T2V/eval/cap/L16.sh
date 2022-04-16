# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/eval_by_cap.py --input_modality video --output_modality text --get_text_desc --use_manual_annotation --output_dir output/TextVideoGPT_L16 --cap_ckpt_file checkpoints/generation/V2T/VideoTextGPT_L32/epoch=54-step=599999.ckpt
