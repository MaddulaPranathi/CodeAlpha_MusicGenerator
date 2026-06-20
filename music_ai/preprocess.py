import argparse
import json
import os
from glob import glob

import numpy as np
from tqdm import tqdm

from midi_utils import midi_to_note_tokens


def build_vocab(token_lists):
    # Special tokens always included
    vocab = {'PAD': 0, 'SOS': 1, 'EOS': 2}
    for toks in token_lists:
        for t in toks:
            if t not in vocab and not t.startswith('PAD'):
                vocab[t] = len(vocab)
    return vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='data')
    ap.add_argument('--out_dir', type=str, default='artifacts')
    ap.add_argument('--step_ms', type=int, default=100)
    ap.add_argument('--max_files', type=int, default=0, help='0 = all')
    ap.add_argument('--max_notes_per_file', type=int, default=0, help='0 = no limit')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    midi_files = []
    for ext in ('*.mid', '*.midi', '*.MID', '*.MIDI'):
        midi_files.extend(glob(os.path.join(args.data_dir, '**', ext), recursive=True))

    midi_files = sorted(set(midi_files))
    if args.max_files and args.max_files > 0:
        midi_files = midi_files[: args.max_files]

    if not midi_files:
        raise SystemExit(f'No MIDI files found under {args.data_dir}')

    all_tokens = []
    token_lists = []

    for path in tqdm(midi_files, desc='Tokenizing MIDI'):
        max_notes = None if args.max_notes_per_file <= 0 else args.max_notes_per_file
        toks = midi_to_note_tokens(path, step_ms=args.step_ms, max_notes=max_notes)
        if len(toks) < 2:
            continue
        token_lists.append(toks)
        all_tokens.extend(toks)

    if len(all_tokens) < 1000:
        print(f'Warning: only {len(all_tokens)} tokens collected. Training may be poor.')

    vocab = build_vocab(token_lists)
    inv_vocab = {v: k for k, v in vocab.items()}

    tokens_ids = np.array([vocab[t] for t in all_tokens], dtype=np.int64)

    np.save(os.path.join(args.out_dir, 'tokens.npy'), tokens_ids)
    with open(os.path.join(args.out_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    meta = {
        'num_files': len(midi_files),
        'total_tokens': int(tokens_ids.shape[0]),
        'step_ms': args.step_ms,
    }
    with open(os.path.join(args.out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('Done.')
    print('Vocab size:', len(vocab))


if __name__ == '__main__':
    main()

