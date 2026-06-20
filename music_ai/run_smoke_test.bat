@echo off
setlocal

REM Installs dependencies
python -m pip install --upgrade pip
pip install -r music_ai\requirements.txt

REM Create folders
if not exist data mkdir data
if not exist artifacts mkdir artifacts
if not exist outputs mkdir outputs

REM Preprocess (fails fast if no MIDI)
python music_ai\preprocess.py --data_dir data --out_dir artifacts --step_ms 100 --max_files 0

REM Train tiny run for smoke test
python music_ai\train.py --artifacts_dir artifacts --epochs 1 --batch_size 32 --seq_len 128

REM Generate
python music_ai\generate.py --artifacts_dir artifacts --out_midi outputs\generated.mid --length 512 --temperature 1.0

echo Done. Output: outputs\generated.mid
endlocal

