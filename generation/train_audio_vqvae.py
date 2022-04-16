# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------

import _init_path
import sys
import fire
import jukebox.utils.dist_adapter as dist
from models.audio_vqvae.hparams import setup_hparams
from jukebox.make_models import make_vqvae, make_prior
from jukebox.utils.logger import init_logging
from jukebox.utils.torch_utils import count_parameters
from jukebox.utils.dist_utils import print_once
from jukebox.data.data_processor import DataProcessor
from jukebox.train import get_optimizer, get_ema, get_ddp, train, evaluate

def run(hps="teeny", port=29500, **kwargs):
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    hps = setup_hparams(hps, kwargs)
    hps.ngpus = dist.get_world_size()
    hps.argv = " ".join(sys.argv)
    hps.bs_sample = hps.bs

    # Setup dataset
    data_processor = DataProcessor(hps)

    # Setup models
    vqvae = make_vqvae(hps, device)
    print_once(f"Parameters VQVAE:{count_parameters(vqvae)}")
    if hps.prior:
        prior = make_prior(hps, vqvae, device)
        print_once(f"Parameters Prior:{count_parameters(prior)}")
        model = prior
    else:
        model = vqvae

    # Setup opt, ema and distributed_model.
    opt, shd, scalar = get_optimizer(model, hps)
    ema = get_ema(model, hps)
    distributed_model = get_ddp(model, hps)

    logger, metrics = init_logging(hps, local_rank, rank)
    logger.iters = model.step

    # Run training, eval, sample
    for epoch in range(hps.curr_epoch, hps.epochs):
        metrics.reset()
        data_processor.set_epoch(epoch)
        if hps.train:
            train_metrics = train(distributed_model, model, opt, shd, scalar, ema, logger, metrics, data_processor, hps)
            train_metrics['epoch'] = epoch
            if rank == 0:
                print('Train',' '.join([f'{key}: {val:0.4f}' for key,val in train_metrics.items()]))
            dist.barrier()

        if hps.test:
            if ema: ema.swap()
            test_metrics = evaluate(distributed_model, model, logger, metrics, data_processor, hps)
            test_metrics['epoch'] = epoch
            if rank == 0:
                print('Ema',' '.join([f'{key}: {val:0.4f}' for key,val in test_metrics.items()]))
            dist.barrier()
            if ema: ema.swap()
        dist.barrier()

if __name__ == '__main__':
    fire.Fire(run)