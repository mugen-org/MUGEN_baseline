# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/eval_by_sim.py --input_modality video --output_modality text --output_source gt --clip_ckpt_file checkpoints/retrieval/video_text_retrieval/epoch=15.pt --get_game_frame --get_text_desc --use_manual_annotation --output_dir output/Sim/VT_GT