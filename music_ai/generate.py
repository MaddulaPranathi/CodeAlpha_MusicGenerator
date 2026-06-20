import argparse
import json
import os

import numpy as np
import torch

from model import LSTMNextToken
from midi_utils import tokens_to_midi


def sample_next_token(logits, temperature: float = 1.0):
    logits = logits / max(1e-8, temperature)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts_dir', type=str, default='artifacts')
    ap.add_argument('--checkpoint', type=str, default=os.path.join('artifacts', 'checkpoints', 'model.pt'))
    ap.add_argument('--out_midi', type=str, default='outputs/generated.mid')
    ap.add_argument('--length', type=int, default=2048)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--program', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    vocab = ckpt['vocab']
    inv_vocab = {int(v): k for k, v in vocab.items()}
    vocab_size = ckpt['vocab_size']

    cfg = ckpt['config']
    model = LSTMNextToken(
        vocab_size=vocab_size,
        emb_dim=cfg.get('emb_dim', 128),
        hidden_dim=cfg.get('hidden_dim', 256),
        num_layers=cfg.get('num_layers', 2),
        dropout=cfg.get('dropout', 0.2),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    step_ms = 100
    meta_path = os.path.join(args.artifacts_dir, 'meta.json')
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, 'r', encoding='utf-8'))
        step_ms = int(meta.get('step_ms', step_ms))

    sos_id = vocab.get('SOS', 1)
    eos_id = vocab.get('EOS', 2)

    seq = [sos_id]
    context = torch.tensor([sos_id], dtype=torch.long, device=device).unsqueeze(0)  # (1,1)

    with torch.no_grad():
        for _ in range(args.length - 1):
            # For efficiency, feed last seq_len tokens if present
            logits = model(context)  # (1, T, V)
            next_logits = logits[0, -1, :]
            next_id = sample_next_token(next_logits, temperature=args.temperature)
            seq.append(next_id)
            if next_id == eos_id:
                break

            # Keep a small context window
            context = torch.tensor([seq[-cfg.get('seq_len', 256):]], dtype=torch.long, device=device)

    os.makedirs(os.path.dirname(args.out_midi) or '.', exist_ok=True)
    tokens_to_midi(seq, inv_vocab, args.out_midi, step_ms=step_ms, program=args.program)
    print('Saved:', args.out_midi)


if __name__ == '__main__':
    main()

