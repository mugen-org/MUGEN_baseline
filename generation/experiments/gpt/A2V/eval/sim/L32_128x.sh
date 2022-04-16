# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/eval_by_sim.py --input_modality audio --output_modality video --clip_ckpt_file checkpoints/retrieval/video_audio_retrieval/epoch=16.pt --get_audio --output_dir output/AudioVideoGPT_L32_128x