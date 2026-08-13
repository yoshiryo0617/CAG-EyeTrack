python src/preprocess/build_analysis_table.py \
    --features_root data/with_features \
    --save_path data/gd_analysis.csv.gz \
    --models cag rnng plm action_lstm transformer lstm \
    --seeds 3435 3436 3437
