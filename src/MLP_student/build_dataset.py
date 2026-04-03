import torch
from torch.utils.data import Dataset

class DistillDataset(Dataset):
    def __init__(self, id2idx, hgnn_embeddings, clip_dict):
        self.data = []

        for item_id, idx in id2idx.items():
            if item_id in clip_dict:
                clip = clip_dict[item_id]
                hgnn = hgnn_embeddings[idx]

                self.data.append((clip, hgnn))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip, hgnn = self.data[idx]
        return clip.float(), hgnn.float()