"""Derive the remaining word-level predictors used by the regression models.

Operates on the output of harmonize_surprisals.py, adding screen position, spillover terms
for the preceding words, out-of-vocabulary flags, sentence boundary flags and the length of
the sentence each word belongs to.
"""

import argparse
import os
import re

import pandas as pd

TASKS = ("task1- SR", "task2 - NR")

ALPHABET = re.compile(r"^[a-zA-Z]+$")

# Task 1 was presented over two sessions of 200 sentences each.
TASK1_SESSION_SIZE = 200


def add_screen_n(df, task):
    if task == "task1- SR":
        first_session = df[df["sent_id"] <= TASK1_SESSION_SIZE - 1]["sent_id"] + 1
        second_session = df[df["sent_id"] > TASK1_SESSION_SIZE - 1]["sent_id"] - (TASK1_SESSION_SIZE - 1)
        df["screen_N"] = pd.concat([first_session, second_session], axis=0)
        return df

    df["screen_N"] = df["sent_id"] + 1
    return df


def add_is_first(df):
    df["is_first"] = df["segment_N"] == 1
    return df


def add_is_last(df):
    df["is_last"] = df["is_first"].shift(-1, fill_value=True)
    return df


def add_is_second_last(df):
    df["is_second_last"] = df["is_last"].shift(-1, fill_value=False)
    return df


def add_prev_length(df):
    df["prev_length"] = df["length"].shift(1, fill_value=0)
    df["prev2_length"] = df["length"].shift(2, fill_value=0)
    df["prev3_length"] = df["length"].shift(3, fill_value=0)
    return df


def add_prev_freq(df):
    df["prev_freq"] = df["freq"].shift(1, fill_value=0)
    df["prev2_freq"] = df["freq"].shift(2, fill_value=0)
    df["prev3_freq"] = df["freq"].shift(3, fill_value=0)
    return df


def add_next_freq(df):
    df["next_freq"] = df["freq"].shift(-1, fill_value=0)
    return df


def add_prev_surp(df):
    df["prev_surp"] = df["surp"].shift(1, fill_value=0)
    df["prev2_surp"] = df["surp"].shift(2, fill_value=0)
    df["prev3_surp"] = df["surp"].shift(3, fill_value=0)
    return df


def is_alphabetical(text):
    return bool(re.match(ALPHABET, text))


def add_alphabetical(df):
    return df.assign(alphabetical=[is_alphabetical(word) for word in df["word"].values])


def add_prev_alphabetical(df):
    df["prev_alphabetical"] = df["alphabetical"].shift(1, fill_value=False)
    df["prev2_alphabetical"] = df["alphabetical"].shift(2, fill_value=False)
    df["prev3_alphabetical"] = df["alphabetical"].shift(3, fill_value=False)
    return df


def add_is_oov(df):
    df["oov"] = df["freq"] == 0
    df["prev_oov"] = df["oov"].shift(1, fill_value=False)
    return df


def add_is_sent_first(df):
    df["is_sent_first"] = df["word_id"] == 0
    return df


def add_is_sent_last(df):
    df["is_sent_last"] = df["is_first"].shift(-1, fill_value=True)
    return df


def add_max_word_id(df):
    max_word_id = df.groupby(["sent_id"])["word_id"].max()
    max_word_id.name = "max_word_id"
    return df.join(max_word_id, on="sent_id")


def add_features(df, task):
    df = add_screen_n(df, task)
    df = add_is_first(df)
    df = add_is_last(df)
    df = add_is_second_last(df)
    df = add_prev_length(df)
    df = add_prev_freq(df)
    df = add_next_freq(df)
    df = add_prev_surp(df)
    df = add_alphabetical(df)
    df = add_prev_alphabetical(df)
    df = add_is_oov(df)
    df = add_is_sent_first(df)
    df = add_is_sent_last(df)
    df = add_max_word_id(df)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonized_root", required=True, help="Output directory of harmonize_surprisals.py.")
    parser.add_argument("--save_root", required=True, help="Directory to write the augmented CSVs to.")
    args = parser.parse_args()

    for task in TASKS:
        task_root = os.path.join(args.harmonized_root, task)

        for subject in sorted(os.listdir(task_root)):
            subject_root = os.path.join(task_root, subject)
            if not os.path.isdir(subject_root):
                continue

            print(subject_root)

            for model in sorted(os.listdir(subject_root)):
                model_root = os.path.join(subject_root, model)
                if not os.path.isdir(model_root):
                    continue

                save_root = os.path.join(args.save_root, task, subject, model)
                os.makedirs(save_root, exist_ok=True)

                for csv_name in sorted(os.listdir(model_root)):
                    if not csv_name.endswith(".csv"):
                        continue
                    df = pd.read_csv(os.path.join(model_root, csv_name))
                    add_features(df, task).to_csv(os.path.join(save_root, csv_name), index=False)


if __name__ == "__main__":
    main()
