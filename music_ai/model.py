import torch
import torch.nn as nn


class LSTMNextToken(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h0=None, c0=None):
        # x: (B, T)
        emb = self.embedding(x)  # (B, T, E)
        if h0 is not None and c0 is not None:
            out, (hn, cn) = self.lstm(emb, (h0, c0))
        else:
            out, (hn, cn) = self.lstm(emb)
        out = self.dropout(out)
        logits = self.fc(out)  # (B, T, V)
        return logits

