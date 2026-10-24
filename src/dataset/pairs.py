import torch
from torch.utils.data import Dataset


class EntityAlignmentDataset(Dataset):
    def __init__(self, pairs: torch.Tensor):
        self.set_pairs(pairs)

    def set_pairs(self, pairs: torch.Tensor):
        if not isinstance(pairs, torch.Tensor):
            raise TypeError("pairs must be a torch.Tensor")
        self.pairs = pairs

    def append_pairs(self, pairs: torch.Tensor):
        if pairs.numel() == 0:
            return
        self.pairs = torch.cat((self.pairs, pairs), dim=0)

    def __len__(self):
        return int(self.pairs.size(0))

    def __getitem__(self, index):
        return self.pairs[index]
