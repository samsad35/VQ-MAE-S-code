import torch
import numpy as np
from einops import repeat, rearrange
import math
import random


def mosaic_indexes(size: int):
    forward_indexes = np.arange(size).reshape(int(math.sqrt(size)), int(math.sqrt(size)))
    rand = random.randint(0, 1)
    if rand == 0:
        a = forward_indexes[:, 0::2]
        b = forward_indexes[:, 1::2]
        for i in range(16):
            if i % 2 != 0:
                temps = a[i].copy()
                a[i] = b[i]
                b[i] = temps
            else:
                pass
        forward_indexes = np.concatenate((a, b)).reshape(-1)
    else:
        a = forward_indexes[:, 1::2]
        b = forward_indexes[:, 0::2]
        for i in range(16):
            if i % 2 != 0:
                temps = a[i].copy()
                a[i] = b[i]
                b[i] = temps
            else:
                pass
        forward_indexes = np.concatenate((a, b)).reshape(-1)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffleMosaic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - 0.50))
        indexes = [mosaic_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)
        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]
        return patches, forward_indexes, backward_indexes


if __name__ == '__main__':
    masking = PatchShuffleMosaic()
    w = torch.randn((256, 1, 128))
    patches, forward_indexes, backward_indexes = masking(w)
    print(forward_indexes.shape)
    print(backward_indexes.shape)
