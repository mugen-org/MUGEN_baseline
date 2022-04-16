import random
random.seed(1)
import torch

all_paths = {}
all_paths["V2T"] = {
    "gt": "/checkpoint/songyangzhang/mugen/Sim/VT_GT/sim/similarities_0_-1.th",
    "pred": [
        "/checkpoint/songyangzhang/mugen/VideoTextGPT_L8/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/VideoTextGPT_L16/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/VideoTextGPT_L32/sim/similarities_0_-1.th",
    ],
}

all_paths["A2T"] = {
    "gt": "/checkpoint/songyangzhang/mugen/Sim/AT_GT/sim/similarities_0_-1.th",
    "pred": [
        "/checkpoint/songyangzhang/mugen/AudioTextGPT_1024x/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/AudioTextGPT_512x/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/AudioTextGPT_256x/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/AudioTextGPT_128x/sim/similarities_0_-1.th",
    ],
}

all_paths["T2V"] = {
    "gt": "/checkpoint/songyangzhang/mugen/Sim/VT_GT/sim/similarities_0_-1.th",
    "pred": [
        "/checkpoint/songyangzhang/mugen/TextVideoGPT_L8/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/TextVideoGPT_L16/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/TextVideoGPT_L32/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/TextVideoGPT_L32_A/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/TextVideoGPT_L32_M+A/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/TextVideoGPT_L8_Down/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/TextVideoGPT_L16_Down/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/TextVideoGPT_L32_Down/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/TextVideoGPT_L32_Down_A/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/TextVideoGPT_L32_Down_M+A/sim/similarities_0_-1.th",
    ],
}

all_paths["A2V"] = {
    "gt": "/checkpoint/songyangzhang/mugen/Sim/AV_GT/sim/similarities_0_-1.th",
    "pred": [
        "/checkpoint/songyangzhang/mugen/AudioVideoGPT_L32_1024x/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/AudioVideoGPT_L32_512x/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/AudioVideoGPT_L32_256x/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/AudioVideoGPT_L32_128x/sim/similarities_0_-1.th",
    ],
}

all_paths["V2A"] = {
    "gt": "/checkpoint/songyangzhang/mugen/Sim/AV_GT/sim/similarities_0_-1.th",
    "pred": [
        "/checkpoint/songyangzhang/mugen/VideoAudioGPT_L32_1024x/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/VideoAudioGPT_L32_512x/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/VideoAudioGPT_L32_256x/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/VideoAudioGPT_L32_128x/sim/similarities_0_-1.th",
    ],
}

all_paths["T2A"] = {
    "gt": "/checkpoint/songyangzhang/mugen/Sim/AT_GT/sim/similarities_0_-1.th",
    "pred": [
        "/checkpoint/songyangzhang/mugen/TextAudioGPT_1024x/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/TextAudioGPT_512x/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/TextAudioGPT_256x/sim/similarities_0_-1.th",
        "/checkpoint/songyangzhang/mugen/TextAudioGPT_128x/sim/similarities_0_-1.th",
    ],
}



def r_sim_scores(root_paths):
    for root_path in root_paths["pred"]:
        pred_score = torch.load(root_path)
        gt_score = torch.load(root_paths['gt'])[:,:1]
        print("pred score: %.1f gt score: %.1f ratio: %.1f"%(torch.mean(pred_score), torch.mean(gt_score), torch.mean(pred_score)/torch.mean(gt_score)*100))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, choices=['V2T', 'A2T', 'T2V', 'A2V', 'V2A', 'T2A'])
    args = parser.parse_args()
    root_paths = all_paths[args.task]
    r_sim_scores(root_paths)