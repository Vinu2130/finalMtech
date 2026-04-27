python scripts/01_prepare_data.py --output_dir data/processed --coin_n 8000 --llc_n 8000
python scripts/02_build_splits.py --input_dir data/processed --output_dir data/splits

# Baseline start: direct on coin_flip
python scripts/03_train_baseline.py --task coin_flip --method direct --train_file data/splits/coin_flip/train.jsonl --valid_file data/splits/coin_flip/valid_iid.jsonl --output_dir outputs/coin_flip_direct
