python src/preprocess/harmonize_surprisals.py \
    --gaze_root data/zuco_gd \
    --surprisal_root data/surprisals \
    --save_root data/harmonized \
    --models cag rnng plm action_lstm transformer lstm \
    --seeds 3435 3436 3437
