# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
import soundfile
import torchvision.io

import _init_path
import argparse
import torch
from tqdm import tqdm
from data.mugen_data import MUGENDataset
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from models.gpt.gpt import MMGPT
from einops import rearrange
import torch.nn.functional as F
import json
import os
torch.manual_seed(1)

class CaptionEvaluation():
    def __init__(self, args):
        self.args = args
        if args.cap_ckpt_file is not None:
            self.caption_model = MMGPT.load_from_checkpoint(args.cap_ckpt_file).cuda()
            self.caption_model.eval()
        self.dataset = MUGENDataset(self.args, split=self.args.split)

    def eval(self):
        start_idx, end_idx = self.args.start_idx, min(self.args.end_idx, len(self.dataset)) if self.args.end_idx > 0 else len(self.dataset)

        print('setting up captioning scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        res_texts, gt_texts = {}, {}
        sample_idx = 0
        if self.args.cap_ckpt_file is None:
            for i in tqdm(range(start_idx, end_idx)):
                gt_texts.update({i: [self.dataset[i]['text']]})
                sample_path = os.path.join(self.args.output_dir, f"samples", self.dataset.data[i]['video']['id'], f'samples.json')
                sample = json.load(open(sample_path))
                res_texts.update({i: [sample[0]]})

        else:
            for item_idx in tqdm(range(start_idx, end_idx, self.args.batch_size)):
                samples, sample_idxs = [], []
                for i in range(item_idx, min(end_idx,item_idx + self.args.batch_size)):
                    gt_texts.update({i: [self.dataset[i]['text']]})
                    sample_path = os.path.join(self.args.output_dir, f"samples", self.dataset.data[i]['video']['id'], f'samples.json')
                    if os.path.exists(sample_path):
                        res_texts.update({i: json.load(open(sample_path))})
                        continue
                    if self.args.input_modality == 'audio':
                        sample_path = os.path.join(self.args.output_dir, f"samples", self.dataset.data[i]['video']['id'], f'sample_{sample_idx}.wav')
                        x = torch.from_numpy(soundfile.read(sample_path)[0]) - 0.5
                        x = F.pad(x, (0, int(self.args.audio_sample_length) - x.shape[0]))[None,:,None]
                        # x = F.interpolate(x[None, None], size=[70560]).permute(0, 2, 1)

                    elif self.args.input_modality == 'video':
                        sample_path = os.path.join(self.args.output_dir, f"samples", self.dataset.data[i]['video']['id'], f'sample_{sample_idx}.mp4')
                        x = torchvision.io.read_video(sample_path)[0][None]
                        B, T, H, W, C = x.shape
                        if x.shape[0] != self.args.sequence_length:
                            x = F.interpolate(rearrange(x, "b t h w c -> b c t h w"), [self.args.sequence_length] + [H, W])
                            x = rearrange(x, "b c t h w -> b t h w c")
                    samples.append(x)
                    sample_idxs.append(i)
                if len(samples) > 0:
                    batch = {self.args.input_modality: torch.cat(samples, dim=0).cuda()}
                    samples = self.caption_model.sample(len(samples), batch, top_k=self.args.caption_top_k, top_p=self.args.caption_top_p)
                    for j, sample in zip(sample_idxs, samples):
                        sample_path = os.path.join(self.args.output_dir, f"samples", self.dataset.data[j]['video']['id'], f'samples.json')
                        json.dump([sample], open(sample_path, 'w'))
                        res_texts.update({j: [sample]})

        # compute the coco caption metrics
        metrics = {}
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gt_texts, res_texts)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f"%(m, sc))
                    metrics[m] = sc
            else:
                print("%s: %0.3f"%(method, score))
                metrics[method] = score
        output_dir = os.path.join(self.args.output_dir, "cap")
        os.makedirs(output_dir, exist_ok=True)
        json.dump(metrics, open(os.path.join(output_dir, "scores.json"), 'w'))

    @staticmethod
    def add_eval_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--cap_ckpt_file", type=str, default=None)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--start_idx", type=int, default=0)
        parser.add_argument("--end_idx", type=int, default=-1)
        parser.add_argument("--caption_top_k", type=int, default=1)
        parser.add_argument("--caption_top_p", type=float, default=0.5)
        parser.add_argument("--input_modality", type=str, default=None, choices=['video', 'audio'])
        parser.add_argument("--output_modality", type=str, default=None, choices=['text'])
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--split", type=str, default="test")
        return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = MUGENDataset.add_data_specific_args(parser)
    parser = CaptionEvaluation.add_eval_specific_args(parser)
    args = parser.parse_args()

    evaluation = CaptionEvaluation(args)
    evaluation.eval()