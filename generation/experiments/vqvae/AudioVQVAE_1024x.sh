# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
python generation/train_audio_vqvae.py --hps=vqvae_coinrun_1024x_full_mix --name=AudioVQVAE_1024x --bs=4 \
--sample_length=65536 --nworkers=4 --audio_files_dir=datasets/coinrun/mix_audio \
--labels=False --train --local_logdir output/