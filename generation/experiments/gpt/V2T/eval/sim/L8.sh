# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/eval_by_sim.py --input_modality video --output_modality text --clip_ckpt_file checkpoints/retrieval/video_text_retrieval/epoch=15.pt --get_game_frame --output_dir output/VideoTextGPT_L8