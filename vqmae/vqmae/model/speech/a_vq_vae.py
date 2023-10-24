import torch
import torch.nn as nn
from .Encoder_Decoder import Encoder, Decoder
from .Vector_Quantizer import VectorQuantizer
from .Vector_Quantizer_EMA import VectorQuantizerEMA
from vector_quantize_pytorch import VectorQuantize


class SpeechVQVAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(SpeechVQVAE, self).__init__()

        self._encoder = Encoder(1, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity

    def load(self, path_model: str):
        checkpoint = torch.load(path_model)
        self.load_state_dict(checkpoint['model'])
        loss = checkpoint['loss']
        print(f"\t [Model Audio-VQ-VAE is loaded successfully with loss = {loss}]")

    def get_codebook_indices(self, audio):
        z = self._encoder(audio)
        z = self._pre_vq_conv(z)
        indices = self._vq_vae.get_codebook_indices(z)
        return indices

    def decode(self, indices):
        audio_embeds = self._vq_vae.quantify(indices)
        audio = self._decoder(audio_embeds)
        return audio.squeeze(1)
