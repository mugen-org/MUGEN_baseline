# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------

import _init_path
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from data.mugen_data import MUGENDataset
from data.data import VideoData
from models.gpt.gpt import MMGPT
import os

def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MMGPT.add_model_specific_args(parser)
    parser = MUGENDataset.add_data_specific_args(parser)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pretrained_model', type=str, default=None)
    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.val_dataloader()

    if args.pretrained_model is not None:
        model = MMGPT.load_from_checkpoint(args.pretrained_model, strict=False)  # attention mask is different
        model.args = args  # overwrite arguments
    else:
        model = MMGPT(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=-1))
    callbacks.append(ModelCheckpoint(monitor='val/loss', mode='min', filename='latest_checkpoint'))

    kwargs = dict()
    if args.gpus > 1:
        # find_unused_parameters = False to support gradient checkpointing
        kwargs = dict(distributed_backend='ddp', gpus=args.gpus,
                      plugins=[pl.plugins.DDPPlugin(find_unused_parameters=False)])

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,
                                            max_steps=args.max_steps, **kwargs)

    trainer.fit(model, data)

if __name__ == '__main__':
    main()

