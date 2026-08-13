"""Collapse the per-subject, per-model CSVs into the single table the R analysis reads.

Every model and seed produces the same rows and the same baseline predictors; only the
surprisal columns differ. This script keeps one copy of the shared columns and adds a
`surp_{model}_seed{seed}` / `prev_surp_{model}_seed{seed}` pair per language model, then
drops the rows without a recorded gaze duration.
"""

import argparse
import os

import pandas as pd

TASKS = ("task1- SR", "task2 - NR")

TASK2LABEL = {
    "task1- SR": "task1",
    "task2 - NR": "task2",
}

DEFAULT_MODELS = ("cag", "rnng", "plm", "action_lstm", "transformer", "lstm")
DEFAULT_SEEDS = ("3435", "3436", "3437")

# Column names in the R script use no underscore inside a model name.
MODEL_ALIASES = {"action_lstm": "actionlstm"}

# Columns that differ between models; everything else is stored once.
MODEL_SPECIFIC_COLUMNS = ("surp", "prev_surp", "prev2_surp", "prev3_surp")

# Columns identifying a row, used to check that the models really are row aligned.
KEY_COLUMNS = ["task", "subj", "sent_id", "word_id"]


def alias(model):
    return MODEL_ALIASES.get(model, model)


def load_model_seed(features_root, model, seed):
    """Concatenate every subject of both tasks for one model and seed."""
    frames = []

    for task in TASKS:
        task_root = os.path.join(features_root, task)

        for subject in sorted(os.listdir(task_root)):
            csv_path = os.path.join(task_root, subject, model, f"{seed}.csv")
            if not os.path.isfile(csv_path):
                continue

            df = pd.read_csv(csv_path)
            df["subj"] = subject.split("_")[0]
            df["task"] = TASK2LABEL[task]
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"no CSVs found for model {model!r}, seed {seed!r} under {features_root!r}")

    return pd.concat(frames, axis=0).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_root", required=True, help="Output directory of add_features.py.")
    parser.add_argument("--save_path", required=True, help="Destination CSV; '.gz' enables compression.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    table = None
    keys = None

    for model in args.models:
        for seed in args.seeds:
            df = load_model_seed(args.features_root, model, seed)
            print(f"{model} seed {seed}: {len(df)} rows")

            if table is None:
                keys = df[KEY_COLUMNS]
                shared = [column for column in df.columns if column not in MODEL_SPECIFIC_COLUMNS]
                table = df[shared].copy()
            elif not df[KEY_COLUMNS].equals(keys):
                raise ValueError(f"{model} seed {seed} is not row aligned with the first model")

            table[f"surp_{alias(model)}_seed{seed}"] = df["surp"].values
            table[f"prev_surp_{alias(model)}_seed{seed}"] = df["prev_surp"].values

    table = table[table["GD"].notna()].reset_index(drop=True)
    print(f"{len(table)} rows with a gaze duration, {len(table.columns)} columns")

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    # %.10g keeps more significant digits than the single precision surprisals carry, so
    # the round trip through text is lossless for them.
    table.to_csv(args.save_path, index=False, float_format="%.10g")


if __name__ == "__main__":
    main()
