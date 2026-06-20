import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    seq_len: int = 256


class NoteSequenceDataset(Dataset):
    """Sliding-window dataset for next-token prediction."""

    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens = tokens.astype(np.int64)
        self.seq_len = seq_len
        if len(self.tokens) < seq_len + 1:
            raise ValueError(f"Not enough tokens ({len(self.tokens)}) for seq_len={seq_len}")

    def __len__(self) -> int:
        return len(self.tokens) - (self.seq_len + 1) + 1

    def __getitem__(self, idx: int):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return torch.from_numpy(x), torch.from_numpy(y)


def load_artifacts(artifacts_dir: str) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str], Dict]:
    """Load preprocessed token stream + vocab mappings."""
    vocab_path = os.path.join(artifacts_dir, "vocab.json")
    data_path = os.path.join(artifacts_dir, "tokens.npy")
    meta_path = os.path.join(artifacts_dir, "meta.json")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    inv_vocab = {int(k): v for k, v in json.load(open(vocab_path, "r", encoding="utf-8")).items()}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    tokens = np.load(data_path)
    return tokens, vocab, inv_vocab, meta

