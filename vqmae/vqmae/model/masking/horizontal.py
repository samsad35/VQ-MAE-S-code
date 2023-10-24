import torch
import numpy as np
from einops import repeat, rearrange
import math
import random


def horizontal_indexes(size: int):
    forward_indexes = np.arange(size).reshape(20, 16)
    lines = np.arange(16)
    np.random.shuffle(lines)
    forward_indexes = forward_indexes[:, lines].T.reshape(-1)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffleHorizontal(torch.nn.Module):
    def __init__(self, ratio: float = 0.50) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))-1
        indexes = [horizontal_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


if __name__ == '__main__':
    masking = PatchShuffleHorizontal()
    w = torch.randn((320, 1, 64))
    patches, forward_indexes, backward_indexes = masking(w)
    print(patches.shape)
    print(backward_indexes.shape)

