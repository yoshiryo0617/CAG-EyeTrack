"""Align language model surprisals with the ZuCo word tokens.

The language models tokenise the stimuli differently from ZuCo, so a single ZuCo word can
span several model tokens. For each ZuCo word this script consumes model tokens until
their concatenation matches the word, and assigns the summed surprisal to that word.

Two surprisal file conventions are in use:

* ``cag`` and ``rnng`` write ``surprisals_task{N}_seed{S}.txt``: headerless, tab
  separated, six columns, zero indexed, with two trailing summary lines.
* the remaining models write ``{model}_surprisals_task{N}_seed{S}.txt``: tab separated
  with a header row, one indexed.
"""

import argparse
import os

import pandas as pd
from nltk.tokenize import TreebankWordTokenizer

TASKS = ("task1- SR", "task2 - NR")

TASK2LABEL = {
    "task1- SR": "task1",
    "task2 - NR": "task2",
}

DEFAULT_MODELS = ("cag", "rnng", "plm", "action_lstm", "transformer", "lstm")
DEFAULT_SEEDS = ("3435", "3436", "3437")

# Models whose surprisal files are headerless and already zero indexed.
BARE_FORMAT_MODELS = frozenset({"cag", "rnng"})

BARE_COL_NAMES = ["sent_id", "word_id", "word", "subwords", "surp", "surps"]

COL2COL = {
    "sentence_id": "sent_id",
    "token_id": "word_id",
    "token": "word",
    "surprisal": "surp",
}

# Sentences to drop, mirroring the `utils.ZuCo` module of the ZuCo release. The entry
# keyed by subject `None` applies to every subject of that task.
SKIPPED_SENT_IDS = {
    ("task1- SR", "ZDN_SR"): set(range(150, 250)) | {399},
    ("task2 - NR", None): {10},
    ("task2 - NR", "ZPH_NR"): set(range(50, 100)),
}


def skipped_sent_ids(task, subject):
    skipped = set(SKIPPED_SENT_IDS.get((task, None), set()))
    skipped |= SKIPPED_SENT_IDS.get((task, subject), set())
    return skipped


def surprisal_path(surprisal_root, model, task, seed):
    label = TASK2LABEL[task]
    if model in BARE_FORMAT_MODELS:
        name = f"surprisals_{label}_seed{seed}.txt"
    else:
        name = f"{model}_surprisals_{label}_seed{seed}.txt"
    return os.path.join(surprisal_root, model, name)


def load_surprisals(path, model):
    if model in BARE_FORMAT_MODELS:
        # The last two lines hold corpus level summary statistics rather than tokens.
        return pd.read_csv(path, sep="\t", names=BARE_COL_NAMES)[:-2]

    df = pd.read_csv(path, sep="\t").rename(columns=COL2COL)
    df["sent_id"] -= 1
    df["word_id"] -= 1
    return df


def harmonize(gaze_df, surp_df, skipped, tokenizer):
    """Return `gaze_df` with a `surp` column summed over the model tokens of each word."""
    sent_ids = surp_df["sent_id"].values
    surp_words = surp_df["word"].values
    surps = surp_df["surp"].values

    harmonized_surps = []
    cursor = 0

    for word_gaze in gaze_df["word"].values:
        word_gaze = "".join(tokenizer.tokenize(word_gaze, convert_parentheses=True))

        harmonized_surp = 0.0
        harmonized_word = ""

        while cursor < len(surp_df):
            sent_id = int(sent_ids[cursor])
            surp_word = surp_words[cursor]
            surp = surps[cursor]
            cursor += 1

            if sent_id in skipped:
                continue

            harmonized_surp += surp
            harmonized_word += surp_word

            if harmonized_word == word_gaze:
                harmonized_surps.append(harmonized_surp)
                break

    return gaze_df.assign(surp=harmonized_surps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaze_root", required=True, help="Output directory of extract_gd.py.")
    parser.add_argument("--surprisal_root", required=True, help="Directory holding one subdirectory per model.")
    parser.add_argument("--save_root", required=True, help="Directory to write the harmonized CSVs to.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    tokenizer = TreebankWordTokenizer()

    for task in TASKS:
        gaze_task_root = os.path.join(args.gaze_root, task)

        for gaze_csv_name in sorted(os.listdir(gaze_task_root)):
            if not gaze_csv_name.endswith(".csv"):
                continue

            subject = gaze_csv_name[: -len(".csv")]
            skipped = skipped_sent_ids(task, subject)

            gaze_df = pd.read_csv(os.path.join(gaze_task_root, gaze_csv_name))
            gaze_df = gaze_df[~gaze_df["sent_id"].isin(SKIPPED_SENT_IDS.get((task, None), set()))]

            print(os.path.join(gaze_task_root, gaze_csv_name))

            for model in args.models:
                save_root = os.path.join(args.save_root, task, subject, model)
                os.makedirs(save_root, exist_ok=True)

                for seed in args.seeds:
                    surp_df = load_surprisals(surprisal_path(args.surprisal_root, model, task, seed), model)
                    harmonized = harmonize(gaze_df, surp_df, skipped, tokenizer)
                    harmonized.to_csv(os.path.join(save_root, f"{seed}.csv"), index=False)


if __name__ == "__main__":
    main()
