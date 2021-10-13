import torch
import torch.nn as nn

from .common import EmbeddingLayer, RelPositionalEncoding
from .transformer import Transformer
from .conformer import Conformer
from .predictors import VarianceAdopter
from .utils import sequence_mask, generate_path


class TTSModel(nn.Module):
    def __init__(self, params):
        super(TTSModel, self).__init__()

        self.emb = EmbeddingLayer(**params.embedding, channels=params.encoder.channels // 3)
        self.relative_pos_emb = RelPositionalEncoding(
            params.encoder.channels,
            params.encoder.dropout
        )
        self.encoder = Conformer(**params.encoder)
        self.variance_adopter = VarianceAdopter(**params.variance_adopter)
        self.decoder = Conformer(**params.decoder)

        self.out_conv = nn.Conv1d(params.decoder.channels, params.n_mel, 1)

        self.post_net = nn.Sequential(
            nn.Conv1d(80, params.decoder.channels, 5, padding=2),
            nn.BatchNorm1d(params.decoder.channels),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(params.decoder.channels, params.decoder.channels, 5, padding=2),
            nn.BatchNorm1d(params.decoder.channels),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(params.decoder.channels, params.decoder.channels, 5, padding=2),
            nn.BatchNorm1d(params.decoder.channels),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(params.decoder.channels, params.decoder.channels, 5, padding=2),
            nn.BatchNorm1d(params.decoder.channels),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(params.decoder.channels, 80, 5, padding=2)
        )

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
        x, pos_emb = self.relative_pos_emb(x)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        y_mask = sequence_mask(y_length).unsqueeze(1).to(x.dtype)

        x = self.encoder(x, pos_emb, x_mask)

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
        x, pos_emb = self.relative_pos_emb(x)
        x = self.decoder(x, pos_emb, y_mask)
        x = self.out_conv(x)
        x *= y_mask

        x_post = x + self.post_net(x)
        x_post *= y_mask

        return x, x_post, (dur_pred, pitch_pred, energy_pred), (x_mask, y_mask)

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

        x = x + self.post_net(x)
        x *= y_mask
        return x, pitch
