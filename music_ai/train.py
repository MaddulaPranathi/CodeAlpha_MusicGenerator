import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import NoteSequenceDataset
from model import LSTMNextToken


def load_vocab(artifacts_dir: str):
    vocab_path = os.path.join(artifacts_dir, 'vocab.json')
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    inv_vocab = {int(v): k for k, v in vocab.items()}
    return vocab, inv_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts_dir', type=str, default='artifacts')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seq_len', type=int, default=256)
    ap.add_argument('--emb_dim', type=int, default=128)
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    tokens = np.load(os.path.join(args.artifacts_dir, 'tokens.npy'))

    vocab, _ = load_vocab(args.artifacts_dir)
    vocab_size = len(vocab)

    ds = NoteSequenceDataset(tokens=tokens, seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = LSTMNextToken(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.get('PAD', 0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(os.path.join(args.artifacts_dir, 'checkpoints'), exist_ok=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(dl, desc=f'Epoch {epoch}/{args.epochs}')
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            logits = model(x)  # (B, T, V)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), y.reshape(B * T))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=total_loss / max(1, n_batches))

        avg_loss = total_loss / max(1, n_batches)
        ckpt_path = os.path.join(args.artifacts_dir, 'checkpoints', 'model.pt')
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'config': vars(args),
                'vocab_size': vocab_size,
            },
            ckpt_path,
        )
        print(f'Avg loss: {avg_loss:.4f} | saved: {ckpt_path}')


if __name__ == '__main__':
    main()

