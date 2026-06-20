# TODO — Music AI (MIDI) LSTM Project

- [x] Create project scaffold (README, requirements)
- [x] Implement preprocessing: `preprocess.py` (MIDI -> token ids + vocab)
- [x] Implement dataset + PyTorch LSTM model
- [x] Implement training script `train.py`
- [x] Implement generation script `generate.py` (tokens -> MIDI)
- [ ] Run a smoke test: `python preprocess.py --data_dir data --out_dir artifacts` (requires user MIDI files)
- [ ] Train on small subset (optional): `--max_files 50`
- [ ] Generate and verify output MIDI in `outputs/`

