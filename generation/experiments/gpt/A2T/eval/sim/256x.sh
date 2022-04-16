# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/eval_by_sim.py --input_modality audio --output_modality text --clip_ckpt_file checkpoints/retrieval/audio_text_retrieval/epoch=17.pt --get_audio --output_dir output/AudioTextGPT_256x