# Part-of-speech distribution of the ZuCo data points before and after data exclusion.
#
# Supplementary analysis reported in the appendix to Chapter 3 of
#   Ryo Yoshida (2027). The Hybrid Approach to Computational Psycholinguistics in the Era
#   of Transformers. Doctoral dissertation, The University of Tokyo.
# It is not part of the CoNLL 2025 paper.
#
# The data exclusion follows modeling_gd.R: words without a first-pass gaze duration,
# sentence-initial and sentence-final words, out-of-vocabulary words (and the words
# following them), and outliers beyond three standard deviations of the residuals of the
# baseline regression model.
#
# Inputs:
#   ../data/zuco_gd/          output of src/preprocess/extract_gd.py (all words, including
#                             those without a gaze duration); see scripts/extract_gd.sh
#   ../data/gd_analysis.csv.gz the bundled analysis table (words with a gaze duration)
# Outputs:
#   results/pos_distribution.csv, and LaTeX table rows printed to the standard output.
#
# Run with this r-workspace directory as the working directory:
#   Rscript pos_distribution.R [path/to/zuco_gd]

library(lme4)

args = commandArgs(trailingOnly = TRUE)
zuco_gd_root = if (length(args) >= 1) args[1] else '../data/zuco_gd'

is_true = function(x) toupper(as.character(x)) == 'TRUE'

# POS tags with fewer occurrences than this in all data points (before exclusion) are
# collapsed into "Other".
MIN_COUNT = 1000

# all words
# Sentence 10 of the natural reading task is dropped for every subject, as in
# src/preprocess/harmonize_surprisals.py.
read_zuco_gd = function(task) {
  files = list.files(file.path(zuco_gd_root, task), pattern = '\\.csv$', full.names = TRUE)
  if (length(files) == 0) stop('No CSV files found under ', file.path(zuco_gd_root, task))
  df = do.call(rbind, lapply(files, function(f) {
    read.csv(f, colClasses = 'character', na.strings = '')[, c('sent_id', 'pos', 'is_content', 'GD')]
  }))
  if (task == 'task2 - NR') df = df[df$sent_id != '10', ]
  df
}
all_words = rbind(read_zuco_gd('task1- SR'), read_zuco_gd('task2 - NR'))
all_words$missing = is.na(all_words$GD)

# words with a gaze duration, and the exclusion of modeling_gd.R
data = read.csv('../data/gd_analysis.csv.gz')
stopifnot(nrow(data) == sum(!all_words$missing))

data$reason = NA_character_
data$reason[is_true(data$is_sent_first) | is_true(data$is_sent_last)] = 'sentence_initial_or_final'
data$reason[is.na(data$reason) & (data$freq <= 0 | data$prev_freq <= 0)] = 'oov'

kept = data[is.na(data$reason), ]
kept$length = scale(kept$length)
kept$prev_length = scale(kept$prev_length)
kept$freq = scale(log2(kept$freq))
kept$prev_freq = scale(log2(kept$prev_freq))
kept$sent_id = scale(kept$sent_id)
kept$word_id = scale(kept$word_id)
kept$subj = as.factor(kept$subj)
baseline = lmer(
  log(GD) ~ length + prev_length + freq + prev_freq + sent_id + word_id + (1 | subj),
  kept,
  REML = FALSE
)
outlier = !(scale(resid(baseline)) < 3.0)
data$reason[which(is.na(data$reason))[outlier]] = 'outlier'
analyzed = data[is.na(data$reason), ]

cat('Data points\n')
cat('  all:', nrow(all_words), '\n')
cat('  missing:', sum(all_words$missing), '\n')
cat('  sentence-initial or sentence-final:', sum(data$reason == 'sentence_initial_or_final', na.rm = TRUE), '\n')
cat('  out-of-vocabulary:', sum(data$reason == 'oov', na.rm = TRUE), '\n')
cat('  before outlier removal:', nrow(kept), '\n')
cat('  outliers:', sum(outlier), '\n')
cat('  analyzed:', nrow(analyzed), '\n\n')

# word class: function words are the tags in FUNCTION_WORD_POS of extract_gd.py
all_words$class = ifelse(is_true(all_words$is_content), 'Content words', 'Function words')
analyzed$class = ifelse(is_true(analyzed$is_content), 'Content words', 'Function words')

cat('Proportion of missing values (first-pass skipping) by word class\n')
for (cl in c('Content words', 'Function words')) {
  x = all_words$missing[all_words$class == cl]
  cat(sprintf('  %s: %.1f%% of %d\n', cl, 100 * mean(x), length(x)))
}
cat('\n')

# table
count_by = function(x, levels) as.vector(table(factor(x, levels = levels)))

tags = names(sort(table(all_words$pos), decreasing = TRUE))
n_all = count_by(all_words$pos, tags)
n_analyzed = count_by(analyzed$pos, tags)
frequent = n_all >= MIN_COUNT

classes = c('Content words', 'Function words')
rows = data.frame(
  label = c(tags[frequent], 'Other', classes, 'Total'),
  group = c(rep('pos', sum(frequent)), 'pos', rep('class', 2), 'total'),
  all = c(n_all[frequent], sum(n_all[!frequent]), count_by(all_words$class, classes), nrow(all_words)),
  analyzed = c(n_analyzed[frequent], sum(n_analyzed[!frequent]), count_by(analyzed$class, classes), nrow(analyzed))
)
rows$all_pct = 100 * rows$all / nrow(all_words)
rows$analyzed_pct = 100 * rows$analyzed / nrow(analyzed)
rows$excluded_pct = 100 * (rows$all - rows$analyzed) / rows$all

dir.create('results', showWarnings = FALSE)
write.csv(rows, 'results/pos_distribution.csv', row.names = FALSE)

cat('Tags collapsed into Other:', paste(tags[!frequent], collapse = ', '), '\n\n')

fmt_n = function(n) formatC(n, format = 'd', big.mark = ',')
fmt_label = function(label, group) {
  if (group != 'pos' || label == 'Other') return(label)
  paste0('\\texttt{', gsub('$', '\\$', label, fixed = TRUE), '}')
}
cat('LaTeX rows\n')
for (i in seq_len(nrow(rows))) {
  if (rows$group[i] != rows$group[max(i - 1, 1)]) cat('\\midrule\n')
  cat(sprintf('%s & %s (%.1f) & %s (%.1f) & %.1f \\\\\n',
              fmt_label(rows$label[i], rows$group[i]),
              fmt_n(rows$all[i]), rows$all_pct[i],
              fmt_n(rows$analyzed[i]), rows$analyzed_pct[i],
              rows$excluded_pct[i]))
}
