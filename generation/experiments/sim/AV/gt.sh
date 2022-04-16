# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/eval_by_sim.py --input_modality video --output_modality audio --output_source gt --clip_ckpt_file checkpoints/retrieval/video_audio_retrieval/epoch=16.pt --get_game_frame --get_audio --output_dir output/Sim/AV_GT
