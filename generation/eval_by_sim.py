import random

import _init_path
import argparse
import json

import soundfile
import torch
import torchvision.io

from models.videoclip.clip import CLIPModel
from data.mugen_data import MUGENDataset
from tqdm import tqdm
# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
import torch.nn.functional as F
from glob import glob
import os

class SimilarityEvaluation():
    def __init__(self, args):
        self.args = args
        self.input_modality = args.input_modality
        self.output_modality = args.output_modality
        self.clip_model = CLIPModel(
            video_enc = "video" in [args.input_modality, args.output_modality],
            audio_enc = "audio" in [args.input_modality, args.output_modality],
            text_enc = "text" in [args.input_modality, args.output_modality],
        ).cuda()
        self.clip_model.load_state_dict(torch.load(args.clip_ckpt_file, map_location=torch.device("cuda")), strict=True)
        self.clip_model.eval()
        self.clip_model.cuda()
        self.dataset = MUGENDataset(args=args, split=args.split)

    def get_pred_batch(self, modality, i):
        save_dir = os.path.join(args.output_dir, f"samples", self.dataset.data[i]['video']['id'])
        batch = {}
        if modality == "video":
            assert len(glob(os.path.join(save_dir, "*.mp4"))) == self.args.num
            batch["video"] = torch.stack([torchvision.io.read_video(video_path)[0] for video_path in glob(os.path.join(save_dir, "*.mp4"))], dim=0).cuda()
        elif modality == "audio":
            assert len(glob(os.path.join(save_dir, "*.wav"))) == self.args.num
            audios = []
            for audio_path in glob(os.path.join(save_dir, "*.wav")):
                x = torch.from_numpy(soundfile.read(audio_path)[0])-0.5
                F.pad(x, (0, int(self.args.audio_sample_length) - x.shape[0]))
                audios.append(x)
            batch["audio"] = torch.stack(audios, dim=0)[...,None].cuda()
        elif modality == "text":
            batch["text"] = json.load(open(os.path.join(save_dir, "samples.json")))
        return batch

    def get_gt_batch(self, modality, i):
        batch = {}
        item = self.dataset[i]
        if modality == "video":
            batch["video"] = item["video"][None].expand(args.num, -1, -1, -1, -1).cuda()
        elif modality == "audio":
            batch["audio"] = item['audio'][None].expand(args.num, -1, -1).cuda()
        elif modality == "text":
            batch["text"] = [item['text'] for _ in range(args.num)]
        return batch

    def get_rand_batch(self, modality, i):
        batch = {}
        j = random.choice(list(range(0,i))+list(range(i+1,len(self.dataset))))
        item = self.dataset[j]
        if modality == "video":
            batch["video"] = item["video"][None].expand(args.num, -1, -1, -1, -1).cuda()
        elif modality == "audio":
            batch["audio"] = item['audio'][None].expand(args.num, -1, -1).cuda()
        elif modality == "text":
            batch["text"] = [item['text'] for _ in range(args.num)]
        return batch

    def eval(self):
        start_idx, end_idx = self.args.start_idx, min(self.args.end_idx, len(self.dataset)) if self.args.end_idx > 0 else len(self.dataset)
        print(f"processing samples from {start_idx} to {end_idx}")
        similarities = []
        for i in tqdm(range(start_idx, end_idx)):
            batch = {}
            batch.update(getattr(self, f"get_{self.args.input_source}_batch")(self.args.input_modality, i))
            batch.update(getattr(self, f"get_{self.args.output_source}_batch")(self.args.output_modality, i))
            with torch.no_grad():
                similarities.append(self.clip_model.compute_similarities(batch, query_type=self.input_modality, output_type=self.output_modality)[0].cpu())
        similarities = torch.stack(similarities, dim=0)
        output_dir = os.path.join(args.output_dir, 'sim')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(similarities, os.path.join(output_dir, f"similarities_{self.args.start_idx}_{self.args.end_idx}.th"))


    @staticmethod
    def add_eval_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--clip_ckpt_file", type=str, required=True)
        parser.add_argument("--start_idx", type=int, default=0)
        parser.add_argument("--end_idx", type=int, default=-1)
        parser.add_argument("--num", type=int, default=5)
        parser.add_argument("--input_modality", type=str, default=None, choices=['video', 'audio', 'text'])
        parser.add_argument("--output_modality", type=str, default=None, choices=['video', 'audio', 'text'])
        parser.add_argument("--input_source", type=str, default='gt', choices=['rand', 'pred', 'gt'])
        parser.add_argument("--output_source", type=str, default='pred', choices=['rand', 'pred', 'gt'])
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--split", type=str, default="test")
        return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = MUGENDataset.add_data_specific_args(parser)
    parser = SimilarityEvaluation.add_eval_specific_args(parser)
    args = parser.parse_args()
    evaluation = SimilarityEvaluation(args)
    evaluation.eval()