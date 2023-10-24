import torch
import torch.nn as nn
from .masked_autoencoder import MAE_Encoder
from einops import rearrange
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


class Query2Label(nn.Module):
    """Modified Query2Label model
    Unlike the model described in the paper (which uses a modified DETR
    transformer), this version uses a standard, unmodified Pytorch Transformer.
    Learnable label embeddings are passed to the decoder module as the target
    sequence (and ultimately is passed as the Query to MHA).
    """

    def __init__(
            self, encoder: MAE_Encoder, num_classes,
            nheads=8,
            encoder_layers=1,
            decoder_layers=2,
            use_pos_encoding=True):
        """Initializes model
        Args:
            model (str): Timm model descriptor for backbone.
            conv_out (int): Backbone output channels.
            num_classes (int): Number of possible label classes
            hidden_dim (int, optional): Hidden channels from linear projection of
            backbone output. Defaults to 256.
            nheads (int, optional): Number of MHA heads. Defaults to 8.
            encoder_layers (int, optional): Number of encoders. Defaults to 6.
            decoder_layers (int, optional): Number of decoders. Defaults to 6.
            use_pos_encoding (bool, optional): Flag for use of position encoding.
            Defaults to False.
        """

        super().__init__()

        self.num_classes = num_classes
        self.use_pos_encoding = use_pos_encoding
        self.proj = encoder.proj.requires_grad_(True)
        self.cls_token = encoder.cls_token
        self.hidden_dim = self.cls_token.shape[-1]
        self.pos_embedding = encoder.pos_embedding
        self.transformer_mae = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.seq_length = encoder.seq_length
        self.dim_indices = encoder.dim_indices
        self.emb_dim = encoder.emb_dim
        self.transformer = nn.Transformer(self.hidden_dim, nheads, encoder_layers, decoder_layers)

        if self.use_pos_encoding:
            # returns the encoding object
            self.pos_encoder = PositionalEncoding1D(self.hidden_dim)
            # returns the summing object
            self.encoding_adder = Summer(self.pos_encoder)

        # prediction head
        self.classifier = nn.Linear(num_classes * self.hidden_dim, num_classes)

        # learnable label embedding
        self.label_emb = nn.Parameter(torch.rand(1, num_classes, self.hidden_dim))

    def forward(self, x):
        """Passes batch through network
        Args:
            x (Tensor): Batch of images
        Returns:
            Tensor: Output of classification head
        """

        patches = rearrange(x, 'b t c -> t b c')
        patches = self.proj(patches).reshape(self.seq_length, -1, self.dim_indices * self.emb_dim)
        patches = patches + self.pos_embedding
        # patches = self.pos_embedding(patches)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer_mae(patches))

        # add position encodings
        if self.use_pos_encoding:
            # input with encoding added
            features = self.encoding_adder(features)

        features = rearrange(features, 'b t c -> t b c')
        B = features.shape[1]

        # image feature vector "h" is sent in after transformation above; we
        # also convert label_emb from [1 x TARGET x (hidden)EMBED_SIZE] to
        # [TARGET x BATCH_SIZE x (hidden)EMBED_SIZE]
        label_emb = self.label_emb.repeat(B, 1, 1)
        label_emb = label_emb.transpose(0, 1)
        h = self.transformer(features, label_emb).transpose(0, 1)

        # output from transformer was of dim [TARGET x BATCH_SIZE x EMBED_SIZE];
        # however, we transposed it to [BATCH_SIZE x TARGET x EMBED_SIZE] above.
        # below we reshape to [BATCH_SIZE x TARGET*EMBED_SIZE].
        #
        # next, we project transformer outputs to class labels
        h = torch.reshape(h, (B, self.num_classes * self.hidden_dim))

        return self.classifier(h)
