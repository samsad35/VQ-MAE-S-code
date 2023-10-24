from .masked_autoencoder import MAE_Encoder
import torch
from einops import rearrange


class Classifier(torch.nn.Module):
    def __init__(self, encoder: MAE_Encoder, num_classes=7):
        super().__init__()
        self.proj = encoder.proj
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.cls_token.shape[-1], num_classes)
        self.seq_length = encoder.seq_length
        self.dim_indices = encoder.dim_indices
        self.emb_dim = encoder.emb_dim

    def forward(self, patches):
        patches = rearrange(patches, 'b t c -> t b c')
        patches = self.proj(patches).reshape(self.seq_length, -1, self.dim_indices * self.emb_dim)
        patches = patches + self.pos_embedding
        # patches = self.pos_embedding(patches)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits

    def get_cls(self, patches):
        patches = rearrange(patches, 'b t c -> t b c')
        patches = self.proj(patches).reshape(self.seq_length, -1, self.dim_indices * self.emb_dim)
        patches = patches + self.pos_embedding
        # patches = self.pos_embedding(patches)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        cls = features[0]
        return cls
