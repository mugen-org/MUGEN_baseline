# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------

import _init_path
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from data.data import VideoData
from data.mugen_data import MUGENDataset
from models.video_vqvae.vqvae import VQVAE

def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--model', type=str, required=True)
    parser = VQVAE.add_model_specific_args(parser)
    parser = MUGENDataset.add_data_specific_args(parser)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.val_dataloader()
    data.test_dataloader()

    # automatically adjust learning rate
    bs, base_lr, ngpu, accumulate = args.batch_size, args.lr, args.gpus, args.accumulate_grad_batches
    args.lr = accumulate * (ngpu/8.) * (bs/4.) * base_lr
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
        args.lr, accumulate, ngpu/8, bs/4, base_lr))

    model = VQVAE(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss', mode='min', save_top_k=-1))
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss', mode='min', filename='latest_checkpoint'))

    kwargs = dict()
    if args.gpus > 1:
        num_nodes = int(os.environ['SLURM_JOB_NUM_NODES']) if 'SLURM_JOB_NUM_NODES' in os.environ else 1
        kwargs = dict(distributed_backend='ddp', gpus=args.gpus//num_nodes, num_nodes=num_nodes)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, max_steps=args.max_steps, **kwargs)

    trainer.fit(model, data)

if __name__ == '__main__':
    main()