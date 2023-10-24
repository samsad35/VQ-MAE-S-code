import torch
import timm
import math
import numpy as np
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from .masking import PatchShuffle, PatchShuffleHorizontal, PatchShuffleVertical, PatchShuffleMosaic
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


def size_model(model: torch.nn.Module):
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x


"""
        MASKED Encoder

"""


class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 seq_length=None,
                 emb_dim=None,
                 num_layer=12,
                 num_head=2,
                 mask_ratio=0.25,
                 dim_input: int = 270,
                 masking: str = "random",  # ["random", "horizontal", "vertical", "mosaic", "half"]
                 trainable_position: bool = True
                 ) -> None:
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        if trainable_position:
            self.pos_embedding = torch.nn.Parameter(torch.zeros(seq_length, 1, emb_dim))
        else:
            self.pos_embedding = PositionalEncoding(d_model=emb_dim, max_len=seq_length)
        #
        if masking.lower() == "random":
            self.shuffle = PatchShuffle(mask_ratio)
        elif masking.lower() == "horizontal":
            self.shuffle = PatchShuffleHorizontal(ratio=mask_ratio)
        elif masking.lower() == "vertical":
            self.shuffle = PatchShuffleVertical(ratio=mask_ratio)
        elif masking.lower() == "mosaic":
            self.shuffle = PatchShuffleMosaic()
        else:
            raise Exception("masking must be random or horizontal, vertical, half or mosaic")
        self.proj = torch.nn.Linear(dim_input, emb_dim)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.emb_dim = emb_dim
        self.seq_length = seq_length
        self.mask_ratio = mask_ratio
        self.trainable_position = trainable_position
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        if self.trainable_position:
            trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, patches):
        patches = rearrange(patches, 'b t c -> t b c')
        patches = self.proj(patches).reshape(self.seq_length, -1, self.emb_dim)
        if self.trainable_position:
            patches = patches + self.pos_embedding
        else:
            patches = self.pos_embedding(patches)
        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        return features, backward_indexes


"""
        MASKED Decoder

"""


class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 seq_length=None,
                 emb_dim=None,
                 num_layer=4,
                 num_head=2,
                 dim_output=270,
                 trainable_position: bool = True
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        if trainable_position:
            self.pos_embedding = torch.nn.Parameter(torch.zeros(seq_length + 1, 1, emb_dim))
        else:
            self.pos_embedding = PositionalEncoding(d_model=emb_dim, max_len=seq_length + 1)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.head = torch.nn.Linear(emb_dim, dim_output)
        self.trainable_position = trainable_position
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        if self.trainable_position:
            trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat(
            [torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat(
            [features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)],
            dim=0)
        features = take_indexes(features, backward_indexes)
        if self.trainable_position:
            features = features + self.pos_embedding
        else:
            features = self.pos_embedding(features)
        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # remove global feature

        features = self.head(features)
        patches = features
        mask = torch.zeros_like(patches)
        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        mask = rearrange(mask, 't b c -> b t c')
        patches = rearrange(patches, 't b c -> b t c')
        return patches, mask


"""
        MASKED AUTOENCODER

"""


class SpecMAE(torch.nn.Module):
    def __init__(self,
                 seq_length=16,
                 emb_dim=64 * 2 * 2,
                 encoder_layer=12,
                 encoder_head=4,
                 decoder_layer=4,
                 decoder_head=4,
                 mask_ratio=0.75,
                 dim_output: int = 270,
                 dim_input: int = 270,
                 masking: str = "random",
                 trainable_position: bool = True
                 ) -> None:
        super().__init__()
        assert dim_input == dim_output, "input dimension is different of output dimension"

        self.encoder = MAE_Encoder(seq_length=seq_length,
                                   emb_dim=emb_dim,
                                   num_layer=encoder_layer,
                                   num_head=encoder_head,
                                   mask_ratio=mask_ratio,
                                   masking=masking,
                                   dim_input=dim_input,
                                   trainable_position=trainable_position)

        self.decoder = MAE_Decoder(seq_length=seq_length,
                                   emb_dim=emb_dim,
                                   num_layer=decoder_layer,
                                   num_head=decoder_head,
                                   dim_output=dim_output,
                                   trainable_position=trainable_position)
        self.mask_ratio = mask_ratio

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask

    def load(self, path_model: str):
        checkpoint = torch.load(path_model)
        state_dict = checkpoint["model"]
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        self.load_state_dict(state_dict)
        loss = checkpoint['loss']
        print(f"\t [Model robustSMAE is loaded successfully with loss = {loss}]")


if __name__ == '__main__':
    spect = torch.randn(1, 320, 270)
    spect = spect.to("cuda")

    encoder = MAE_Encoder(seq_length=320, emb_dim=320, num_layer=1, mask_ratio=0.75)
    encoder = encoder.to('cuda')
    size_model(encoder)

    decoder = MAE_Decoder(seq_length=320, emb_dim=320, dim_output=270, num_layer=1)
    decoder = decoder.to('cuda')
    size_model(decoder)

    features_, backward_indexes_ = encoder(spect)
    print(features_.shape)
    predicted_spec, mask = decoder(features_, backward_indexes_)
    print(mask[mask == 1].shape)
    print(torch.mean((predicted_spec - spect) ** 2 * mask) / encoder.mask_ratio)
