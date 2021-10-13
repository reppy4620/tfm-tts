import torch
import torch.nn as nn

from .common import EmbeddingLayer
from .transformer import Transformer
from .conformer import Conformer
from .predictors import VarianceAdopter
from .utils import sequence_mask, generate_path


class TTSModel(nn.Module):
    def __init__(self, params):
        super(TTSModel, self).__init__()

        self.emb = EmbeddingLayer(**params.embedding, channels=params.encoder.channels // 3)
        self.encoder = Transformer(**params.encoder)
        self.variance_adopter = VarianceAdopter(**params.variance_adopter)
        self.decoder = Conformer(**params.decoder)

        self.out_conv = nn.Conv1d(params.decoder.channels, params.n_mel, 1)

    def forward(
        self,
        phoneme,
        a1,
        f2,
        x_length,
        y_length,
        duration,
        pitch,
        energy
    ):
        x = self.emb(phoneme, a1, f2)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        y_mask = sequence_mask(y_length).unsqueeze(1).to(x.dtype)

        x = self.encoder(x, x_mask)

        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        path = generate_path(duration.squeeze(1), attn_mask.squeeze(1))

        x, (dur_pred, pitch_pred, energy_pred) = self.variance_adopter(
            x,
            x_mask,
            y_mask,
            pitch,
            energy,
            path
        )
        x = self.decoder(x, y_mask)
        x = self.out_conv(x)
        x *= y_mask

        return x, (dur_pred, pitch_pred, energy_pred), (x_mask, y_mask)

    def infer(self, phoneme, a1, f2, x_length):
        x = self.emb(phoneme, a1, f2)
        x, pos_emb = self.relative_pos_emb(x)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        x, pos_emb = self.relative_pos_emb(x)
        x = self.encoder(x, pos_emb, x_mask)

        x, y_mask, pitch = self.variance_adopter.infer(x, x_mask)
        x, pos_emb = self.relative_pos_emb(x)
        x = self.decoder(x, pos_emb, y_mask)
        x = self.out_conv(x)
        x *= y_mask
        return x, pitch
