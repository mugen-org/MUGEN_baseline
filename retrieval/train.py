# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
import _init_path
import os
from tqdm import tqdm
import torch
from data.mugen_data import MUGENDataset
from models.videoclip.clip import CLIPModel
from utils import AvgMeter, get_lr
import argparse
from torch.utils.tensorboard import SummaryWriter

def build_loaders(args, split):
    dataset = MUGENDataset(args, split)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True if split == "train" else False,
        drop_last=True if split == "train" else False
    )
    return dataloader


def train_epoch(model, train_loader, optimizer, writer, n_iter):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        loss, img_acc, text_acc = model(batch)
        writer.add_scalar("Loss/train", loss.item(), n_iter)
        writer.add_scalar("img_acc/train", img_acc.item(), n_iter)
        writer.add_scalar("text_acc/train", text_acc.item(), n_iter)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = batch["video"].size(0)
        loss_meter.update(loss.item(), count)
        n_iter += 1
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter, n_iter


@torch.no_grad()
def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    img_acc_meter = AvgMeter()
    text_acc_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        loss, img_acc, text_acc = model(batch)

        count = batch["video"].size(0) if "video" in batch else batch['audio'].size(0)
        loss_meter.update(loss.item(), count)
        img_acc_meter.update(img_acc.item(), count)
        text_acc_meter.update(text_acc.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter.avg, img_acc_meter.avg, text_acc_meter.avg


def parse_args():
    parser = argparse.ArgumentParser()
    parser = MUGENDataset.add_data_specific_args(parser)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--default_root_dir', type=str, default='saved_checkpoints')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)

    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--max_temperature', type=float, default=100.0)
    parser.add_argument('--video_enc', action='store_true')
    parser.add_argument('--audio_enc', action='store_true')
    parser.add_argument('--text_enc', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--trainable', action='store_true')

    args = parser.parse_args()
    return args

def train():
    args = parse_args()
    writer = SummaryWriter(log_dir=args.default_root_dir, comment=f"--{args.model_name}")
    train_loader = build_loaders(args, "train")
    valid_loader = build_loaders(args, "val")

    model = CLIPModel(video_enc=args.video_enc, text_enc=args.text_enc, audio_enc=args.audio_enc,
                      pretrained=args.pretrained, trainable=args.trainable, temperature=args.temperature,
                      max_temperature=args.max_temperature).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float('inf')
    n_iter = 0
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss, n_iter = train_epoch(model, train_loader, optimizer, writer, n_iter)

        model.eval()
        valid_loss, img_acc, text_acc = valid_epoch(model, valid_loader)

        writer.add_scalar("Loss/val", valid_loss, epoch)
        writer.add_scalar("img_acc/val", img_acc, epoch)
        writer.add_scalar("text_acc/val", text_acc, epoch)
        torch.save(model.state_dict(), os.path.join(args.default_root_dir, f"epoch={epoch}.pt"))

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(args.default_root_dir, "best_checkpoint.pt"))
            print("Saved Best Model!")

    writer.close()


if __name__ == "__main__":
    train()
