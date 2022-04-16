# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/eval_by_sim.py --input_modality audio --output_modality text --output_source gt --clip_ckpt_file checkpoints/retrieval/audio_text_retrieval/epoch=17.pt --get_audio --get_text_desc --use_manual_annotation --output_dir output/Sim/AT_GT