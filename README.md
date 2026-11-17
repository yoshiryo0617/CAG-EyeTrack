# Investigating Psychometric Predictive Power of Syntactic Attention

This repository provides the code and data for the paper [Investigating Psychometric Predictive Power of Syntactic Attention](https://aclanthology.org/2025.conll-1.27/).

> [Investigating Psychometric Predictive Power of Syntactic Attention](https://aclanthology.org/2025.conll-1.27/) <br>
> Ryo Yoshida, Yushi Sugimoto, and Yohei Oseki <br>
> CoNLL 2025

The analysis models first-pass gaze duration from the [ZuCo 1.0](https://osf.io/q3zws/) corpus with surprisal from six language models, and compares their psychometric predictive power (PPP).

## Models

| Label in the paper | Directory name | Description |
| --- | --- | --- |
| LSTM | `lstm` | sequential LSTM language model |
| RNNG-comp | `action_lstm` | RNNG without the composition function |
| RNNG | `rnng` | Recurrent Neural Network Grammar |
| Transformer | `transformer` | sequential Transformer language model |
| CAG-comp | `plm` | CAG without the composition function |
| CAG | `cag` | Composition Attention Grammar |

Each model was trained with three seeds (`3435`, `3436`, `3437`).

## Requirements

- `python==3.9.13`
- `R==4.4.2`

## Installation

```bash
git clone https://github.com/osekilab/CAG-EyeTrack.git
cd CAG-EyeTrack
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the figures

The analysis table `data/gd_analysis.csv.gz` is included in this repository, so the figures
can be reproduced without downloading ZuCo.

```bash
cd r-workspace
R -e 'renv::restore()'
Rscript modeling_gd.R
```

This writes `r-workspace/results/ppp_comparison.png` and
`r-workspace/results/ppp_length_comparison.png`. We recommend using
[RStudio](https://rstudio.com/) if you want to step through the script interactively;
open `r-workspace/r-workspace.Rproj`.

## Rebuilding the analysis table from ZuCo

Only needed if you want to re-run the preprocessing rather than use the bundled table.

### Eye-tracking data preparation

1. Download ZuCo 1.0 from [OSF](https://osf.io/q3zws/).
2. Unpack it into `data/zuco/`, so that `data/zuco/` contains `task1- SR/`, `task2 - NR/`
   and `task_materials/`.

### Surprisals

The surprisals used in the paper are included under `data/surprisals/`, one directory per
model, with two tasks and three seeds each:

- `cag/` and `rnng/` hold `surprisals_task{1,2}_seed{3435,3436,3437}.txt`: tab separated,
  no header, columns `sent_id word_id word subwords surp surps`, zero indexed, with two
  trailing summary lines.
- the other models hold `{model}_surprisals_task{1,2}_seed{3435,3436,3437}.txt`: tab
  separated with a `sentence_id token_id token surprisal` header, one indexed.

The implementations that produced them are released separately:

- CAG: [osekilab/CAG](https://github.com/osekilab/CAG)
- RNNG: [yoshiryo0617/rnng-pytorch](https://github.com/yoshiryo0617/rnng-pytorch)

### Preprocessing

Run the four stages in order from the repository root:

```bash
bash scripts/extract_gd.sh      # ZuCo .mat -> one CSV per subject
bash scripts/harmonize.sh       # align model tokens with ZuCo words
bash scripts/add_features.sh    # spillover, screen position, OOV, sentence length
bash scripts/build_table.sh     # -> data/gd_analysis.csv.gz
```

Word frequency is read from `data/wikitext-2_train_vocab.txt`, taken from
[wilcoxeg/neural-networks-read-times](https://github.com/wilcoxeg/neural-networks-read-times).

## Additional analysis: POS distribution (dissertation)

This analysis is not part of the CoNLL 2025 paper; it is reported in the appendix to
Chapter 3 of the author's doctoral dissertation. `r-workspace/pos_distribution.R`
compares the part-of-speech distribution of all ZuCo data points with that of the data
points used in the final statistical analysis, applying the same data exclusion as
`modeling_gd.R`. Since it counts words without a gaze duration as well, it needs the
output of `scripts/extract_gd.sh` (`data/zuco_gd/`) in addition to the bundled table:

```bash
cd r-workspace
Rscript pos_distribution.R
```

This prints the number of data points removed at each step and the rows of the table,
and writes `r-workspace/results/pos_distribution.csv`.

## The analysis table

`data/gd_analysis.csv.gz` has one row per word token that has a recorded first-pass gaze
duration: 109,357 rows over 12 subjects and two tasks.

| Group | Columns |
| --- | --- |
| Identifiers | `subj`, `task`, `sent_id`, `word_id`, `word` |
| Dependent variable | `GD` (first-pass gaze duration, ms) |
| Lexical | `length`, `freq`, `pos`, `is_content`, `oov`, `alphabetical` |
| Spillover | `prev_length`, `prev2_length`, `prev3_length`, `prev_freq`, `prev2_freq`, `prev3_freq`, `next_freq`, `prev_oov`, `prev_alphabetical`, `prev2_alphabetical`, `prev3_alphabetical` |
| Layout | `line_N`, `segment_N`, `screen_N`, `session_N`, `article_N`, `is_first`, `is_last`, `is_second_last` |
| Sentence | `is_sent_first`, `is_sent_last`, `max_word_id` |
| Surprisal | `surp_{model}_seed{seed}` and `prev_surp_{model}_seed{seed}` for the six models and three seeds |

`freq` is the log geometric mean of the add-one smoothed WikiText-2 frequency of the
word's subtokens. Surprisals are stored unscaled; `modeling_gd.R` centres and scales them,
and averages over seeds, before fitting.

## Repository layout

```
data/
  surprisals/            surprisals of the six models, two tasks, three seeds
  wikitext-2_train_vocab.txt
  gd_analysis.csv.gz     the table modeling_gd.R reads
src/preprocess/          the four preprocessing stages
scripts/                 shell wrappers for the preprocessing stages
r-workspace/
  modeling_gd.R          mixed effects models, likelihood ratio tests and figures
  pos_distribution.R     POS distribution before and after data exclusion (dissertation)
  renv.lock              R package versions
```

## Citation

```bibtex
@inproceedings{yoshida-etal-2025-investigating,
    title = "Investigating Psychometric Predictive Power of Syntactic Attention",
    author = "Yoshida, Ryo  and
      Sugimoto, Yushi  and
      Oseki, Yohei",
    editor = "Boleda, Gemma  and
      Roth, Michael",
    booktitle = "Proceedings of the 29th Conference on Computational Natural Language Learning",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.conll-1.27/",
    doi = "10.18653/v1/2025.conll-1.27",
    pages = "407--418",
    ISBN = "979-8-89176-271-8"
}
```

## License

The code in this repository is released under the MIT License; see [LICENSE](LICENSE).

`data/gd_analysis.csv.gz` contains gaze durations and stimulus text derived from ZuCo 1.0,
which is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). If
you use it, please cite:

> Hollenstein, N., Rotsztejn, J., Troendle, M., Pedroni, A., Zhang, C., and Langer, N. (2018). <br>
> ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading. <br>
> Scientific Data, 5, 180291.

`data/wikitext-2_train_vocab.txt` is taken from
[wilcoxeg/neural-networks-read-times](https://github.com/wilcoxeg/neural-networks-read-times).

## Note

If you want to obtain the trained models, please contact `yoshiryo0617 [at] g.ecc.u-tokyo.ac.jp`.
