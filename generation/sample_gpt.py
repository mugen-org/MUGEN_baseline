# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
import _init_path
import os
import argparse
from data.mugen_data import MUGENDataset
from models.gpt.gpt import MMGPT
from tqdm import tqdm
from torchvision.io import write_video
import soundfile
import json
from glob import glob
import torch

torch.manual_seed(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser = MMGPT.add_model_specific_args(parser)
    parser = MUGENDataset.add_data_specific_args(parser)
    parser.add_argument("--gpt_ckpt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    return args

def save_samples(args):
    gpt_model = MMGPT.load_from_checkpoint(args.gpt_ckpt_file).cuda()
    gpt_model.eval()
    gpt_model.freeze()
    gpt_model.cuda()
    dataset = MUGENDataset(args=args, split=args.split)
    start_idx, end_idx = args.start_idx, min(args.end_idx, len(dataset)) if args.end_idx > 0 else len(dataset)
    print(f"processing samples from {start_idx} to {end_idx}")
    for i in tqdm(range(start_idx, end_idx)):
        save_dir = os.path.join(args.output_dir, f"samples", dataset.data[i]['video']['id'])
        if (len(glob(os.path.join(save_dir, "*.mp4"))) == args.num and args.output_modality == 'video') \
                or (len(glob(os.path.join(save_dir, "*.wav"))) == args.num and args.output_modality == 'audio'):
            continue

        item = dataset[i]
        if args.input_modality == "video":
            batch = {"video": item["video"][None].expand(args.num, -1, -1, -1, -1).cuda()}
        elif args.input_modality == "audio":
            batch = {"audio": item['audio'][None].expand(args.num, -1, -1).cuda()}
        elif args.input_modality == "text":
            batch = {"text": [item['text'] for _ in range(args.num)]}
        samples = gpt_model.sample(args.num, batch, top_k=args.top_k, top_p=args.top_p)

        os.makedirs(save_dir, exist_ok=True)
        if args.output_modality == 'video':
            for j, sample in enumerate(samples.cpu()):
                save_file = os.path.join(save_dir, f"sample_{j}.mp4")
                write_video(save_file, sample, fps=int(samples[j].shape[0] / 3.2))
        if args.output_modality == 'audio':
            for j, sample in enumerate(samples.cpu()):
                save_file = os.path.join(save_dir, f"sample_{j}.wav")
                soundfile.write(save_file, sample+0.5, samplerate=args.audio_sample_rate, format='wav')
        if args.output_modality == 'text':
            save_file = os.path.join(save_dir, f"samples.json")
            json.dump(samples, open(save_file, 'w'))


if __name__ == '__main__':
    args = parse_args()
    save_samples(args)