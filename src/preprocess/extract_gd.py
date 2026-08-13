"""Extract first-pass gaze duration and word-level predictors from the ZuCo 1.0 release.

Reads the per-subject MATLAB files of tasks 1 (SR) and 2 (NR) and writes one CSV per
subject holding, for every word token, the gaze duration together with the lexical and
layout predictors used as baseline terms in the regression models.
"""

import argparse
import math
import os
import re
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import scipy.io as io
from mosestokenizer import MosesTokenizer
from scipy.stats import gmean

TASKS = ("task1- SR", "task2 - NR")

TASK2SENTENCE_FILE = {
    "task1- SR": "sentiment_labels_task1.csv",
    "task2 - NR": "relations_labels_task2.csv",
}

FUNCTION_WORD_POS = {
    "CC",
    "CD",
    "DT",
    "EX",
    "IN",
    "LS",
    "MD",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RP",
    "TO",
    "UH",
    "WDT",
    "WP",
    "WP$",
    "WRB",
}

# Sentence indices of a few subjects are offset with respect to the stimulus list; see
# the `utils.ZuCo` module distributed with the ZuCo release. Keyed by (task, mat file).
SENT_ID_FIXUPS = {
    ("task1- SR", "resultsZDN_SR.mat"): {150: 250, 399: 400},
    ("task2 - NR", "resultsZPH_NR.mat"): {50: 100},
}

# The stimuli were rendered with at most 80 letters or 13 words per line.
MAX_LETTERS_PER_LINE = 80
MAX_WORDS_PER_LINE = 13


def load_word_freq(vocab_path):
    """Read the WikiText-2 training vocabulary as a token -> count mapping."""
    word_freq = Counter()
    with open(vocab_path) as f:
        for line in f:
            token, freq = line.strip().split("\t")
            word_freq[token] = int(freq)
    return word_freq


def calc_freq(word, tokenizer, word_freq):
    """Log geometric mean of the (add-one smoothed) corpus frequency of a word."""
    tokenized_word = tokenizer(word)
    freqs = [word_freq[word.lower()] + 1 for word in tokenized_word]
    gmean_freq = gmean(freqs)
    log_gmean_freq = math.log(gmean_freq)
    return log_gmean_freq


def get_session_n(task, sent_id):
    if task == "task2 - NR":
        return 1
    if sent_id < 200:
        return 1
    return 2


def get_article_n(task, sent_id):
    if task == "task2 - NR":
        return 1
    return 2


def advance_layout(length, line_n, segment_n, letters_per_line, words_per_line):
    """Track which line of the screen a word falls on, and its position within the line."""
    letters_per_line += length
    words_per_line += 1

    if letters_per_line > MAX_LETTERS_PER_LINE or words_per_line > MAX_WORDS_PER_LINE:
        # This word is the first segment of the next line.
        line_n += 1
        segment_n = 1
        letters_per_line = length
        words_per_line = 1
    else:
        segment_n += 1

    return line_n, segment_n, letters_per_line, words_per_line


def gen_word_stat(
    word_data,
    sent_id,
    word_id,
    words,
    poses,
    line_n,
    segment_n,
    letters_per_line,
    words_per_line,
    session_n,
    article_n,
    tokenizer,
    word_freq,
):
    word = words[word_id]

    word_stat = dict()
    word_stat["sent_id"] = sent_id
    word_stat["word_id"] = word_id
    word_stat["word"] = word
    word_stat["length"] = len(word)
    word_stat["freq"] = calc_freq(word, tokenizer, word_freq)

    pos = poses[word_id][1]
    word_stat["pos"] = pos
    word_stat["is_content"] = pos not in FUNCTION_WORD_POS

    word_stat["GD"] = word_data.GD if not isinstance(word_data.GD, np.ndarray) else np.nan

    line_n, segment_n, letters_per_line, words_per_line = advance_layout(
        word_stat["length"], line_n, segment_n, letters_per_line, words_per_line
    )

    word_stat["line_N"] = line_n
    word_stat["segment_N"] = segment_n
    word_stat["session_N"] = session_n
    word_stat["article_N"] = article_n

    return word_stat, line_n, segment_n, letters_per_line, words_per_line


def gen_nan_word_stat(
    sent_id,
    word_id,
    words,
    poses,
    line_n,
    segment_n,
    letters_per_line,
    words_per_line,
    session_n,
    article_n,
    tokenizer,
    word_freq,
):
    """Placeholder record for a sentence the subject has no recorded word data for."""
    word = words[word_id]

    word_stat = dict()
    word_stat["sent_id"] = sent_id
    word_stat["word_id"] = word_id
    word_stat["word"] = word
    word_stat["length"] = len(word)
    word_stat["freq"] = calc_freq(word, tokenizer, word_freq)

    pos = poses[word_id][0]
    word_stat["pos"] = pos
    word_stat["is_content"] = pos not in FUNCTION_WORD_POS

    word_stat["GD"] = np.nan

    # NOTE: unlike gen_word_stat, the original layout bookkeeping does not add this word's
    # length before testing the line break. Kept as is so that the released data matches
    # the data the paper was written from; these rows carry no gaze duration and are
    # dropped before any model is fit.
    if letters_per_line > MAX_LETTERS_PER_LINE or words_per_line > MAX_WORDS_PER_LINE:
        line_n += 1
        segment_n = 1
        letters_per_line = word_stat["length"]
        words_per_line = 1
    else:
        segment_n += 1

    word_stat["line_N"] = line_n
    word_stat["segment_N"] = segment_n
    word_stat["session_N"] = session_n
    word_stat["article_N"] = article_n

    return word_stat, line_n, segment_n, letters_per_line, words_per_line


def gen_sents_stat(sents_data, sents, task, matfile_name, tokenizer, word_freq):
    fixups = SENT_ID_FIXUPS.get((task, matfile_name), {})

    sents_stat = []
    sent_id = 0

    for sent_data in sents_data:
        # Session and article are resolved from the running index before the fixup is
        # applied, as in the original preprocessing.
        session_n = get_session_n(task, sent_id)
        article_n = get_article_n(task, sent_id)

        sent_id = fixups.get(sent_id, sent_id)

        line_n = 1
        segment_n = 0
        letters_per_line = 0
        words_per_line = 0

        words = sents[sent_id].split(" ")
        poses = nltk.pos_tag(words)

        words_data = sent_data.word

        if not isinstance(words_data, np.ndarray):
            for word_id, _ in enumerate(words):
                word_stat, line_n, segment_n, letters_per_line, words_per_line = gen_nan_word_stat(
                    sent_id,
                    word_id,
                    words,
                    poses,
                    line_n,
                    segment_n,
                    letters_per_line,
                    words_per_line,
                    session_n,
                    article_n,
                    tokenizer,
                    word_freq,
                )
                sents_stat.append(word_stat)
            continue

        for word_id, word_data in enumerate(words_data):
            word_stat, line_n, segment_n, letters_per_line, words_per_line = gen_word_stat(
                word_data,
                sent_id,
                word_id,
                words,
                poses,
                line_n,
                segment_n,
                letters_per_line,
                words_per_line,
                session_n,
                article_n,
                tokenizer,
                word_freq,
            )
            sents_stat.append(word_stat)

        sent_id += 1

    return sents_stat


def load_sents(zuco_root, task):
    path = os.path.join(zuco_root, "task_materials", TASK2SENTENCE_FILE[task])
    if task == "task1- SR":
        df = pd.read_csv(path, sep=";")
        df = df.dropna(subset=["sentence"]).reset_index(drop=True)
    else:
        df = pd.read_csv(path)
    return df["sentence"].values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zuco_root",
        required=True,
        help="Directory holding the unpacked ZuCo 1.0 release "
        "(contains 'task1- SR', 'task2 - NR' and 'task_materials').",
    )
    parser.add_argument(
        "--vocab_path",
        required=True,
        help="WikiText-2 training vocabulary, as a tab separated token/count file.",
    )
    parser.add_argument("--save_root", required=True, help="Directory to write the per-subject CSVs to.")
    args = parser.parse_args()

    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")

    word_freq = load_word_freq(args.vocab_path)
    tokenizer = MosesTokenizer("en")

    for task in TASKS:
        sents = load_sents(args.zuco_root, task)

        matfiles_root = os.path.join(args.zuco_root, task, "Preprocessed", "firstLevelResults")
        save_root = os.path.join(args.save_root, task)
        os.makedirs(save_root, exist_ok=True)

        for matfile_name in sorted(os.listdir(matfiles_root)):
            if not re.match(r".+\.mat", matfile_name):
                continue

            matfile_path = os.path.join(matfiles_root, matfile_name)
            print(matfile_path)

            data = io.loadmat(matfile_path, squeeze_me=True, struct_as_record=False)
            sents_stat = gen_sents_stat(data["sentenceData"], sents, task, matfile_name, tokenizer, word_freq)

            subject = re.search(r"results(.+)\.mat", matfile_name).group(1)
            pd.DataFrame(sents_stat).to_csv(os.path.join(save_root, subject + ".csv"), index=False)


if __name__ == "__main__":
    main()
