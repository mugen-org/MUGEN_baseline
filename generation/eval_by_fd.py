# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
import _init_path
import argparse
import torch
import numpy as np
from tqdm import tqdm
import os
from data.mugen_data import MUGENDataset
from models.fvd.fvd import load_fvd_model, frechet_distance
from models.fad.fad import load_fad_model
import torchvision
import soundfile
import torch.nn.functional as F
from einops import rearrange

class FrechetDistanceEvaluation():
    def __init__(self, args):
        self.args = args
        if args.output_modality == "video":
            self.model = load_fvd_model("cuda")
        elif args.output_modality == "audio":
            self.model = load_fad_model("cuda")
        self.dataset = MUGENDataset(self.args, self.args.split)

    def get_fake_sample(self, item_idx, sample_idx):
        if self.args.output_modality == "video":
            video_path = os.path.join(self.args.output_dir, f"samples", self.dataset.data[item_idx]['video']['id'], f'sample_{sample_idx}.mp4')
            samples = torchvision.io.read_video(video_path)[0]
            T, H, W, C = samples.shape
            if samples.shape[0] != self.args.sequence_length:
                samples = F.interpolate(rearrange(samples[None], "b t h w c -> b c t h w"), [self.args.sequence_length] + [H, W])
                samples = rearrange(samples, "b c t h w -> b t h w c")[0]
        elif self.args.output_modality == "audio":
            audio_path = os.path.join(self.args.output_dir, f"samples", self.dataset.data[item_idx]['video']['id'], f'sample_{sample_idx}.wav')
            samples = soundfile.read(audio_path)[0] - 0.5
            samples = torch.from_numpy(samples)[:, None]
            samples = F.pad(samples, (0, 0, 0, int(self.args.audio_sample_length) - samples.shape[0]))
        return samples

    def get_real_sample(self, item_idx):
        item = self.dataset[item_idx]
        return item[self.args.output_modality]

    @torch.no_grad()
    def get_fd_logits(self, batch):
        if self.args.output_modality == 'video':
            batch = self.model.preprocess(batch, target_resolution=(224, 224)).cuda()
            logits = self.model(batch)
        elif self.args.output_modality == 'audio':
            batch = self.model.preprocess(batch, samplerate=22050).cuda()
            logits = self.model(batch)
        return logits

    def eval(self):
        start_idx, end_idx = self.args.start_idx, min(self.args.end_idx, len(self.dataset)) if self.args.end_idx > 0 else len(self.dataset)
        print(f"processing samples from {start_idx} to {end_idx}")
        fds, all_fake_embeddings, all_real_embeddings = [], [], []
        for sample_idx in range(self.args.num):
            fake_embeddings, real_embeddings = [], []
            for item_idx in tqdm(range(start_idx,end_idx,self.args.batch_size)):
                fake = torch.stack([self.get_fake_sample(i, sample_idx) for i in range(item_idx, min(item_idx+self.args.batch_size, end_idx))], dim=0)
                real = torch.stack([self.get_real_sample(i) for i in range(item_idx, min(item_idx+self.args.batch_size, end_idx))], dim=0)
                fake_embeddings.append(self.get_fd_logits(fake))
                real_embeddings.append(self.get_fd_logits(real))
            fake_embeddings = torch.cat(fake_embeddings, dim=0)
            real_embeddings = torch.cat(real_embeddings, dim=0)
            assert fake_embeddings.shape[0] == real_embeddings.shape[0] == end_idx-start_idx
            print(f"compute fd...")
            fd = frechet_distance(fake_embeddings, real_embeddings)
            fds.append(fd.item())
            all_fake_embeddings.append(fake_embeddings)
            all_real_embeddings.append(real_embeddings)
        print('FD: %.2f ± %.2f'%(np.mean(fds), np.std(fds)))
        output_dir = os.path.join(self.args.output_dir, f"fd")
        os.makedirs(output_dir, exist_ok=True)
        print(f"save embeddings...")
        torch.save(torch.stack(all_fake_embeddings, 0).detach().cpu(), os.path.join(output_dir, f'fake-n{self.args.num}-s{self.args.start_idx}-e{self.args.end_idx}.th'))
        torch.save(torch.stack(all_real_embeddings, 0).detach().cpu(), os.path.join(output_dir, f'real-n{self.args.num}-s{self.args.start_idx}-e{self.args.end_idx}.th'))

    @staticmethod
    def add_eval_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument("--start_idx", type=int, default=0)
        parser.add_argument("--end_idx", type=int, default=-1)
        parser.add_argument("--num", type=int, default=5)
        parser.add_argument("--input_modality", type=str, default=None, choices=['video', 'audio', 'text'])
        parser.add_argument("--output_modality", type=str, default=None, choices=['video', 'audio', 'text'])
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--split", type=str, default="test")
        return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = MUGENDataset.add_data_specific_args(parser)
    parser = FrechetDistanceEvaluation.add_eval_specific_args(parser)
    args = parser.parse_args()

    evaluation = FrechetDistanceEvaluation(args)
    evaluation.eval()