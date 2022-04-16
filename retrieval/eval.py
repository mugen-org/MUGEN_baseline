# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
import torch
from train import build_loaders, parse_args
from models.videoclip.clip import CLIPModel
from tqdm import tqdm

def get_embeddings(model, valid_loader):
    valid_video_embeddings = []
    valid_audio_embeddings = []
    valid_text_embeddings = []

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            if model.video_enc:
                batch['video'] = batch['video'].cuda()
                image_embed = model.get_video_embedding(batch)
                valid_video_embeddings.append(image_embed)
            if model.audio_enc:
                batch['audio'] = batch['audio'].cuda()
                audio_embed = model.get_audio_embedding(batch)
                valid_audio_embeddings.append(audio_embed)
            if model.text_enc:
                caption_embed = model.get_text_embedding(batch)
                valid_text_embeddings.append(caption_embed)

    embeddings = {}
    if model.video_enc: embeddings.update({'video': torch.cat(valid_video_embeddings)})
    if model.audio_enc: embeddings.update({'audio': torch.cat(valid_audio_embeddings)})
    if model.text_enc: embeddings.update({'text': torch.cat(valid_text_embeddings)})
    return embeddings

def compute_recall(similarity_matrix, key_1, key_2):
    top_k = 10
    N = similarity_matrix.shape[0]
    indices = torch.topk(similarity_matrix, dim=0, k=top_k)[1]
    gt_indices = torch.arange(0, N)[None].expand(top_k, -1).cuda()
    recall_1 = torch.sum(gt_indices[:1] == indices[:1]).item() / N
    recall_5 = torch.sum(gt_indices[:5] == indices[:5]).item() / N
    recall_10 = torch.sum(gt_indices[:top_k] == indices[:top_k]).item() / N
    print(f"{key_1}(corpus) {key_2}(query) R@1: {recall_1*100:.2f}% R@5: {recall_5*100:.2f}% R@10: {recall_10*100:.2f}%")

    indices = torch.topk(similarity_matrix, dim=1, k=top_k)[1]
    gt_indices = torch.arange(0, N)[:, None].expand(-1, top_k).cuda()
    recall_1_r = torch.sum(gt_indices[:, :1] == indices[:, :1]).item() / N
    recall_5_r = torch.sum(gt_indices[:, :5] == indices[:, :5]).item() / N
    recall_10_r = torch.sum(gt_indices[:, :top_k] == indices[:, :top_k]).item() / N
    print(f"{key_2}(corpus) {key_1}(query) R@1: {recall_1_r*100:.2f}% R@5: {recall_5_r*100:.2f}% R@10: {recall_10_r*100:.2f}%")
    return recall_1, recall_1_r

@torch.no_grad()
def test_epoch(model, test_loader):
    embeddings = get_embeddings(model, test_loader)

    if model.video_enc and model.text_enc and model.audio_enc:
        key_1, key_2, key_3 = 'video', 'audio', 'text'
        similarity_matrix = (embeddings[key_1] @ embeddings[key_2].T)*torch.exp(model.temperature_va) \
                            + (embeddings[key_1] @ embeddings[key_3].T)*torch.exp(model.temperature_vt)
        recall_1, recall_r = compute_recall(similarity_matrix, key_1, f"{key_2}+{key_3}")
        similarity_matrix = (embeddings[key_2] @ embeddings[key_1].T)*torch.exp(model.temperature_va) \
                            + embeddings[key_2] @ embeddings[key_3].T*torch.exp(model.temperature_at)
        compute_recall(similarity_matrix, key_2, f"{key_1}+{key_3}")
        similarity_matrix = (embeddings[key_3] @ embeddings[key_1].T)*torch.exp(model.temperature_vt) \
                            + (embeddings[key_3] @ embeddings[key_2].T)*torch.exp(model.temperature_at)
        compute_recall(similarity_matrix, key_3, f"{key_1}+{key_2}")
        return recall_1, recall_r
    elif model.video_enc and model.text_enc:
        key_1, key_2 = 'video', 'text'
        similarity_matrix = embeddings[key_1] @ embeddings[key_2].T
        return compute_recall(similarity_matrix, key_1, key_2)
    elif model.video_enc and model.audio_enc:
        key_1, key_2 = 'video', 'audio'
        similarity_matrix = embeddings[key_1] @ embeddings[key_2].T
        return compute_recall(similarity_matrix, key_1, key_2)
    elif model.audio_enc and model.text_enc:
        key_1, key_2 = 'audio', 'text'
        similarity_matrix = embeddings[key_1] @ embeddings[key_2].T
        return compute_recall(similarity_matrix, key_1, key_2)
    else:
        raise NotImplementedError

def eval_emsemble():
    args = parse_args()
    test_loader = build_loaders(args, 'test')

    video_audio_model = CLIPModel(audio_enc=True, video_enc=True, text_enc=False).cuda()
    video_audio_model.load_state_dict(torch.load("checkpoints/retrieval/video_audio_retrieval/epoch=16.pt", map_location="cuda"))
    video_audio_model.eval()

    video_text_model = CLIPModel(audio_enc=False, video_enc=True, text_enc=True).cuda()
    video_text_model.load_state_dict(torch.load("checkpoints/retrieval/video_text_retrieval/epoch=15.pt", map_location="cuda"))
    video_text_model.eval()

    audio_text_model = CLIPModel(audio_enc=True, text_enc=True, video_enc=False).cuda()
    audio_text_model.load_state_dict(torch.load("checkpoints/retrieval/audio_text_retrieval/epoch=17.pt", map_location="cuda"))
    audio_text_model.eval()

    embeddings_va = get_embeddings(video_audio_model, test_loader)
    embeddings_vt = get_embeddings(video_text_model, test_loader)
    embeddings_at = get_embeddings(audio_text_model, test_loader)

    similarity_matrix = (embeddings_va['video'] @ embeddings_va['audio'].T) * torch.exp(video_audio_model.temperature) \
                        + (embeddings_vt['video'] @ embeddings_vt['text'].T) * torch.exp(video_text_model.temperature)
    compute_recall(similarity_matrix, 'video', "audio+text")

    similarity_matrix = (embeddings_va['audio'] @ embeddings_va['video'].T) * torch.exp(video_audio_model.temperature) \
                        + (embeddings_at['audio'] @ embeddings_at['text'].T) * torch.exp(audio_text_model.temperature)
    compute_recall(similarity_matrix, 'audio', "video+text")

    similarity_matrix = (embeddings_vt['text'] @ embeddings_vt['video'].T) * torch.exp(video_text_model.temperature) \
                        + (embeddings_at['text'] @ embeddings_at['audio'].T) * torch.exp(audio_text_model.temperature)
    compute_recall(similarity_matrix, 'text', "video+audio")

def eval_single():
    args = parse_args()
    test_loader = build_loaders(args, 'test')

    video_audio_model = CLIPModel(audio_enc=True, video_enc=True, text_enc=False).cuda()
    video_audio_model.load_state_dict(torch.load("checkpoints/retrieval/video_audio_retrieval/epoch=16.pt", map_location='cuda'))
    video_audio_model.eval()
    test_epoch(video_audio_model, test_loader)


    video_text_model = CLIPModel(audio_enc=False, video_enc=True, text_enc=True).cuda()
    video_text_model.load_state_dict(torch.load("checkpoints/retrieval/video_text_retrieval/epoch=15.pt", map_location='cuda'))
    video_text_model.eval()
    test_epoch(video_text_model, test_loader)

    audio_text_model = CLIPModel(audio_enc=True, text_enc=True, video_enc=False).cuda()
    audio_text_model.load_state_dict(torch.load("checkpoints/retrieval/audio_text_retrieval/epoch=17.pt", map_location='cuda'))
    audio_text_model.eval()
    test_epoch(audio_text_model, test_loader)

    video_audio_text_model = CLIPModel(audio_enc=True, text_enc=True, video_enc=True).cuda()
    video_audio_text_model.load_state_dict(torch.load("checkpoints/retrieval/video_audio_text_retrieval/epoch=17.pt", map_location='cuda'))
    video_audio_text_model.eval()
    test_epoch(video_audio_text_model, test_loader)

if __name__ == '__main__':
    eval_single()
    eval_emsemble()