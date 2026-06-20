# Music AI (MIDI) — LSTM Generation with PyTorch

This project trains an LSTM model on MIDI note sequences and generates new music, then saves the result back to MIDI.

## What it does
1. **Collect MIDI data** (you provide MIDI files; script can scan folders)
2. **Preprocess** MIDI into fixed-time note sequences (using `music21`)
3. **Train** an LSTM language model over tokenized notes
4. **Generate** new token sequences
5. **Convert to MIDI** and save an output `.mid`

## Quickstart
### 1) Create environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install music21 mido numpy tqdm
```

> If you don’t have CUDA, you can install CPU-only PyTorch from the official selector: https://pytorch.org/get-started/locally/

### 3) Put MIDI files in `data/`
Example:
- `data/classical/*.mid`
- `data/jazz/*.mid`

### 4) Preprocess
```bash
python preprocess.py --data_dir data --out_dir artifacts
```

### 5) Train
```bash
python train.py --artifacts_dir artifacts --epochs 20 --batch_size 64
```

### 6) Generate MIDI
```bash
python generate.py --artifacts_dir artifacts --checkpoint artifacts/checkpoints/model.pt --out_midi outputs/generated.mid --length 2048 --temperature 1.0
```

## Expected folder structure
```text
music_ai/
  preprocess.py
  train.py
  generate.py
  dataset.py
  model.py
  midi_utils.py
  requirements.txt
  data/                 (put your MIDI files here)
  artifacts/            (preprocessed arrays, vocab, checkpoints)
  outputs/             (generated MIDI)
  README.md
```

## Notes / limitations
- This is a **baseline** LSTM approach. Music quality will improve with:
  - better tokenization (durations, chords, bar structure)
  - larger datasets
  - more advanced models (Transformer, Music Transformer, etc.)
- Preprocessing here uses a simple fixed grid (`--step_ms`) and encodes `NoteOn` events as tokens.

