# Modeling first-pass gaze duration in ZuCo with surprisal from six language models.
#
# Reproduces the figures of
#   Ryo Yoshida, Yushi Sugimoto and Yohei Oseki (2025).
#   Investigating Psychometric Predictive Power of Syntactic Attention. CoNLL 2025.
#
# Run with this r-workspace directory as the working directory.

renv::restore()
library(lme4)
library(lmerTest)
library(coefplot)
library(e1071)
library(trafo)
library(ggplot2)
library(dplyr)

# loading predictors
# One row per word token with a recorded gaze duration. The surprisals of every model and
# seed are already columns of this table; see src/preprocess/build_analysis_table.py.
data = read.csv('../data/gd_analysis.csv.gz')

# scaling 
## scaling baseline features
data$length = scale(data$length)
data$prev_length = scale(data$prev_length)

data$is_last = as.factor(data$is_last)
data$is_first = as.factor(data$is_first)

data$sent_id = scale(data$sent_id)
data$word_id = scale(data$word_id)

data$subj = as.factor(data$subj)

## scaling surps
data$surp_cag_ave = (data$surp_cag_seed3435 + data$surp_cag_seed3436 + data$surp_cag_seed3437) / 3
data$surp_cag_seed3435 = scale(data$surp_cag_seed3435)
data$surp_cag_seed3436 = scale(data$surp_cag_seed3436)
data$surp_cag_seed3437 = scale(data$surp_cag_seed3437)
data$surp_cag_ave = scale(data$surp_cag_ave)

data$surp_rnng_ave = (data$surp_rnng_seed3435 + data$surp_rnng_seed3436 + data$surp_rnng_seed3437) / 3
data$surp_rnng_seed3435 = scale(data$surp_rnng_seed3435)
data$surp_rnng_seed3436 = scale(data$surp_rnng_seed3436)
data$surp_rnng_seed3437 = scale(data$surp_rnng_seed3437)
data$surp_rnng_ave = scale(data$surp_rnng_ave)

data$surp_plm_ave = (data$surp_plm_seed3435 + data$surp_plm_seed3436 + data$surp_plm_seed3437) / 3
data$surp_plm_seed3435 = scale(data$surp_plm_seed3435)
data$surp_plm_seed3436 = scale(data$surp_plm_seed3436)
data$surp_plm_seed3437 = scale(data$surp_plm_seed3437)
data$surp_plm_ave = scale(data$surp_plm_ave)

data$surp_actionlstm_ave = (data$surp_actionlstm_seed3435 + data$surp_actionlstm_seed3436 + data$surp_actionlstm_seed3437) / 3
data$surp_actionlstm_seed3435 = scale(data$surp_actionlstm_seed3435)
data$surp_actionlstm_seed3436 = scale(data$surp_actionlstm_seed3436)
data$surp_actionlstm_seed3437 = scale(data$surp_actionlstm_seed3437)
data$surp_actionlstm_ave = scale(data$surp_actionlstm_ave)

data$surp_transformer_ave = (data$surp_transformer_seed3435 + data$surp_transformer_seed3436 + data$surp_transformer_seed3437) / 3
data$surp_transformer_seed3435 = scale(data$surp_transformer_seed3435)
data$surp_transformer_seed3436 = scale(data$surp_transformer_seed3436)
data$surp_transformer_seed3437 = scale(data$surp_transformer_seed3437)
data$surp_transformer_ave = scale(data$surp_transformer_ave)

data$surp_lstm_ave = (data$surp_lstm_seed3435 + data$surp_lstm_seed3436 + data$surp_lstm_seed3437) / 3
data$surp_lstm_seed3435 = scale(data$surp_lstm_seed3435)
data$surp_lstm_seed3436 = scale(data$surp_lstm_seed3436)
data$surp_lstm_seed3437 = scale(data$surp_lstm_seed3437)
data$surp_lstm_ave = scale(data$surp_lstm_ave)

## scaling prev surp
data$prev_surp_cag_ave = (data$prev_surp_cag_seed3435 + data$prev_surp_cag_seed3436 + data$prev_surp_cag_seed3437) / 3
data$prev_surp_cag_seed3435 = scale(data$prev_surp_cag_seed3435)
data$prev_surp_cag_seed3436 = scale(data$prev_surp_cag_seed3436)
data$prev_surp_cag_seed3437 = scale(data$prev_surp_cag_seed3437)
data$prev_surp_cag_ave = scale(data$prev_surp_cag_ave)

data$prev_surp_rnng_ave = (data$prev_surp_rnng_seed3435 + data$prev_surp_rnng_seed3436 + data$prev_surp_rnng_seed3437) / 3
data$prev_surp_rnng_seed3435 = scale(data$prev_surp_rnng_seed3435)
data$prev_surp_rnng_seed3436 = scale(data$prev_surp_rnng_seed3436)
data$prev_surp_rnng_seed3437 = scale(data$prev_surp_rnng_seed3437)
data$prev_surp_rnng_ave = scale(data$prev_surp_rnng_ave)

data$prev_surp_plm_ave = (data$prev_surp_plm_seed3435 + data$prev_surp_plm_seed3436 + data$prev_surp_plm_seed3437) / 3
data$prev_surp_plm_seed3435 = scale(data$prev_surp_plm_seed3435)
data$prev_surp_plm_seed3436 = scale(data$prev_surp_plm_seed3436)
data$prev_surp_plm_seed3437 = scale(data$prev_surp_plm_seed3437)
data$prev_surp_plm_ave = scale(data$prev_surp_plm_ave)

data$prev_surp_actionlstm_ave = (data$prev_surp_actionlstm_seed3435 + data$prev_surp_actionlstm_seed3436 + data$prev_surp_actionlstm_seed3437) / 3
data$prev_surp_actionlstm_seed3435 = scale(data$prev_surp_actionlstm_seed3435)
data$prev_surp_actionlstm_seed3436 = scale(data$prev_surp_actionlstm_seed3436)
data$prev_surp_actionlstm_seed3437 = scale(data$prev_surp_actionlstm_seed3437)
data$prev_surp_actionlstm_ave = scale(data$prev_surp_actionlstm_ave)

data$prev_surp_transformer_ave = (data$prev_surp_transformer_seed3435 + data$prev_surp_transformer_seed3436 + data$prev_surp_transformer_seed3437) / 3
data$prev_surp_transformer_seed3435 = scale(data$prev_surp_transformer_seed3435)
data$prev_surp_transformer_seed3436 = scale(data$prev_surp_transformer_seed3436)
data$prev_surp_transformer_seed3437 = scale(data$prev_surp_transformer_seed3437)
data$prev_surp_transformer_ave = scale(data$prev_surp_transformer_ave)

data$prev_surp_lstm_ave = (data$prev_surp_lstm_seed3435 + data$prev_surp_lstm_seed3436 + data$prev_surp_lstm_seed3437) / 3
data$prev_surp_lstm_seed3435 = scale(data$prev_surp_lstm_seed3435)
data$prev_surp_lstm_seed3436 = scale(data$prev_surp_lstm_seed3436)
data$prev_surp_lstm_seed3437 = scale(data$prev_surp_lstm_seed3437)
data$prev_surp_lstm_ave = scale(data$prev_surp_lstm_ave)


# remove
data = data[data$freq > 0, ]
data = data[data$prev_freq > 0, ]

data$freq = scale(log2(data$freq))
data$prev_freq = scale(log2(data$prev_freq))

data = data[data$is_sent_first=="False", ]
data = data[data$is_sent_last=="False", ]

# hist (exploratory; skipped in non-interactive runs)
if (interactive()) {
  hist(log(data$GD))
  dev.off()
  kurtosis(log(data$GD))

  hist(data$length)
  hist(data$prev_length)

  hist(data$freq)
  hist(data$prev_freq)

  hist(data$screen_N)
  hist(data$line_N)
  hist(data$segment_N)

  hist(data$sent_id)
  hist(data$word_id)
}

# fitting baseline
control = lmerControl(optCtrl=list(maxfun=100000))
GD_lmer = lmer(
  log(GD) ~ length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  data,
  REML = FALSE
)
summary(GD_lmer)
coefplot(GD_lmer)

# trimming outlier
dataTrimWhole = data[scale(resid(GD_lmer)) < 3.0, ]
nrow(data) - nrow(dataTrimWhole)
GD_lmer = lmer(
  log(GD) ~ length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer)
coefplot(GD_lmer)

# fitting
## add cag surp
GD_lmer_cag_seed3435 = lmer(
  log(GD) ~ surp_cag_seed3435 + prev_surp_cag_seed3435 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_cag_seed3435)

GD_lmer_cag_seed3436 = lmer(
  log(GD) ~ surp_cag_seed3436 + prev_surp_cag_seed3436 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_cag_seed3436)

GD_lmer_cag_seed3437 = lmer(
  log(GD) ~ surp_cag_seed3437 + prev_surp_cag_seed3437 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_cag_seed3437)

GD_lmer_cag_ave = lmer(
  log(GD) ~ surp_cag_ave + prev_surp_cag_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_cag_ave)


## add rnng surp
GD_lmer_rnng_seed3435 = lmer(
  log(GD) ~ surp_rnng_seed3435 + prev_surp_rnng_seed3435 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_rnng_seed3435)

GD_lmer_rnng_seed3436 = lmer(
  log(GD) ~ surp_rnng_seed3436 + prev_surp_rnng_seed3436 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_rnng_seed3436)

GD_lmer_rnng_seed3437 = lmer(
  log(GD) ~ surp_rnng_seed3437 + prev_surp_rnng_seed3437 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_rnng_seed3437)

GD_lmer_rnng_ave = lmer(
  log(GD) ~ surp_rnng_ave + prev_surp_rnng_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_rnng_seed3437)


## add plm surp
GD_lmer_plm_seed3435 = lmer(
  log(GD) ~ surp_plm_seed3435 + prev_surp_plm_seed3435 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_plm_seed3435)

GD_lmer_plm_seed3436 = lmer(
  log(GD) ~ surp_plm_seed3436 + prev_surp_plm_seed3436 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_plm_seed3436)

GD_lmer_plm_seed3437 = lmer(
  log(GD) ~ surp_plm_seed3437 + prev_surp_plm_seed3437 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_plm_seed3437)

GD_lmer_plm_ave = lmer(
  log(GD) ~ surp_plm_ave + prev_surp_plm_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_plm_seed3437)

## add actionlstm surp
GD_lmer_actionlstm_seed3435 = lmer(
  log(GD) ~ surp_actionlstm_seed3435 + prev_surp_actionlstm_seed3435 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_actionlstm_seed3435)

GD_lmer_actionlstm_seed3436 = lmer(
  log(GD) ~ surp_actionlstm_seed3436 + prev_surp_actionlstm_seed3436 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_actionlstm_seed3436)

GD_lmer_actionlstm_seed3437 = lmer(
  log(GD) ~ surp_actionlstm_seed3437 + prev_surp_actionlstm_seed3437 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_actionlstm_seed3437)

GD_lmer_actionlstm_ave = lmer(
  log(GD) ~ surp_actionlstm_ave + prev_surp_actionlstm_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_actionlstm_ave)


## add transformer surp
GD_lmer_transformer_seed3435 = lmer(
  log(GD) ~ surp_transformer_seed3435 + prev_surp_transformer_seed3435 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_transformer_seed3435)

GD_lmer_transformer_seed3436 = lmer(
  log(GD) ~ surp_transformer_seed3436 + prev_surp_transformer_seed3436 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_transformer_seed3436)

GD_lmer_transformer_seed3437 = lmer(
  log(GD) ~ surp_transformer_seed3437 + prev_surp_transformer_seed3437 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_transformer_seed3437)

GD_lmer_transformer_ave = lmer(
  log(GD) ~ surp_transformer_ave + prev_surp_transformer_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_transformer_ave)


## add lstm surp
GD_lmer_lstm_seed3435 = lmer(
  log(GD) ~ surp_lstm_seed3435 + prev_surp_lstm_seed3435 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_lstm_seed3435)

GD_lmer_lstm_seed3436 = lmer(
  log(GD) ~ surp_lstm_seed3436 + prev_surp_lstm_seed3436 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_lstm_seed3436)

GD_lmer_lstm_seed3437 = lmer(
  log(GD) ~ surp_lstm_seed3437 + prev_surp_lstm_seed3437 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_lstm_seed3437)

GD_lmer_lstm_ave = lmer(
  log(GD) ~ surp_lstm_ave + prev_surp_lstm_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_lstm_ave)

# anova
## anova cag
anova(GD_lmer, GD_lmer_cag_seed3435)
loglik_cag_3435 = logLik(GD_lmer_cag_seed3435) - logLik(GD_lmer)

anova(GD_lmer, GD_lmer_cag_seed3436)
loglik_cag_3436 = logLik(GD_lmer_cag_seed3436) - logLik(GD_lmer) 

anova(GD_lmer, GD_lmer_cag_seed3437)
loglik_cag_3437 = logLik(GD_lmer_cag_seed3437) - logLik(GD_lmer) 

loglik_cag = c(loglik_cag_3435,loglik_cag_3436,loglik_cag_3437)
print(loglik_cag)
mean(loglik_cag)
sd(loglik_cag)

anova(GD_lmer, GD_lmer_cag_ave)
loglik_cag_ave = logLik(GD_lmer_cag_ave) - logLik(GD_lmer)
print(loglik_cag_ave)


## anova rnng
anova(GD_lmer, GD_lmer_rnng_seed3435)
loglik_rnng_3435 = logLik(GD_lmer_rnng_seed3435) - logLik(GD_lmer) 

anova(GD_lmer, GD_lmer_rnng_seed3436)
loglik_rnng_3436 = logLik(GD_lmer_rnng_seed3436) - logLik(GD_lmer) 

anova(GD_lmer, GD_lmer_rnng_seed3437)
loglik_rnng_3437 = logLik(GD_lmer_rnng_seed3437) - logLik(GD_lmer)

loglik_rnng = c(loglik_rnng_3435,loglik_rnng_3436,loglik_rnng_3437)
print(loglik_rnng)
mean(loglik_rnng)
sd(loglik_rnng)

anova(GD_lmer, GD_lmer_rnng_ave)
loglik_rnng_ave = logLik(GD_lmer_rnng_ave) - logLik(GD_lmer) 
print(loglik_rnng_ave)

## anova plm
anova(GD_lmer, GD_lmer_plm_seed3435)
loglik_plm_3435 = logLik(GD_lmer_plm_seed3435) - logLik(GD_lmer)

anova(GD_lmer, GD_lmer_plm_seed3436)
loglik_plm_3436 = logLik(GD_lmer_plm_seed3436) - logLik(GD_lmer)

anova(GD_lmer, GD_lmer_plm_seed3437)
loglik_plm_3437 = logLik(GD_lmer_plm_seed3437) - logLik(GD_lmer)

loglik_plm = c(loglik_plm_3435,loglik_plm_3436,loglik_plm_3437)
print(loglik_plm)
mean(loglik_plm)
sd(loglik_plm)

loglik_plm_ave = logLik(GD_lmer_plm_ave) - logLik(GD_lmer)
print(loglik_plm_ave)

## anova actionlstm
anova(GD_lmer, GD_lmer_actionlstm_seed3435)
loglik_actionlstm_3435 = logLik(GD_lmer_actionlstm_seed3435) - logLik(GD_lmer)

anova(GD_lmer, GD_lmer_actionlstm_seed3436)
loglik_actionlstm_3436 = logLik(GD_lmer_actionlstm_seed3436) - logLik(GD_lmer)

anova(GD_lmer, GD_lmer_actionlstm_seed3437)
loglik_actionlstm_3437 = logLik(GD_lmer_actionlstm_seed3437) - logLik(GD_lmer) 

loglik_actionlstm = c(loglik_actionlstm_3435,loglik_actionlstm_3436,loglik_actionlstm_3437)
print(loglik_actionlstm)
mean(loglik_actionlstm)
sd(loglik_actionlstm)

loglik_actionlstm_ave = logLik(GD_lmer_actionlstm_ave) - logLik(GD_lmer)
print(loglik_actionlstm_ave)

## anova transformer
anova(GD_lmer, GD_lmer_transformer_seed3435)
loglik_transformer_3435 = logLik(GD_lmer_transformer_seed3435) - logLik(GD_lmer)

anova(GD_lmer, GD_lmer_transformer_seed3436)
loglik_transformer_3436 = logLik(GD_lmer_transformer_seed3436) - logLik(GD_lmer)

anova(GD_lmer, GD_lmer_transformer_seed3437)
loglik_transformer_3437 = logLik(GD_lmer_transformer_seed3437) - logLik(GD_lmer)

loglik_transformer = c(loglik_transformer_3435,loglik_transformer_3436,loglik_transformer_3437)
print(loglik_transformer)
mean(loglik_transformer)
sd(loglik_transformer)

loglik_transformer_ave = logLik(GD_lmer_transformer_ave) - logLik(GD_lmer)
print(loglik_transformer_ave)

## anova lstm
anova(GD_lmer, GD_lmer_lstm_seed3435)
loglik_lstm_3435 = logLik(GD_lmer_lstm_seed3435) - logLik(GD_lmer)

anova(GD_lmer, GD_lmer_lstm_seed3436)
loglik_lstm_3436 = logLik(GD_lmer_lstm_seed3436) - logLik(GD_lmer)

anova(GD_lmer, GD_lmer_lstm_seed3437)
loglik_lstm_3437 = logLik(GD_lmer_lstm_seed3437) - logLik(GD_lmer)

loglik_lstm = c(loglik_lstm_3435,loglik_lstm_3436,loglik_lstm_3437)
print(loglik_lstm)
mean(loglik_lstm)
sd(loglik_lstm)

loglik_lstm_ave = logLik(GD_lmer_lstm_ave) - logLik(GD_lmer)
print(loglik_lstm_ave)

# plot
n_datapoints <- nrow(dataTrimWhole)

loglik_lstm_values = c(loglik_lstm_3435, loglik_lstm_3436, loglik_lstm_3437) / n_datapoints
loglik_actionlstm_values = c(loglik_actionlstm_3435, loglik_actionlstm_3436, loglik_actionlstm_3437) / n_datapoints
loglik_rnng_values = c(loglik_rnng_3435, loglik_rnng_3436, loglik_rnng_3437) / n_datapoints
loglik_transformer_values = c(loglik_transformer_3435, loglik_transformer_3436, loglik_transformer_3437) / n_datapoints
loglik_plm_values = c(loglik_plm_3435, loglik_plm_3436, loglik_plm_3437) / n_datapoints
loglik_cag_values = c(loglik_cag_3435, loglik_cag_3436, loglik_cag_3437) / n_datapoints

plot_data <- data.frame(
  Model = c("LSTM", "RNNG-comp", "RNNG", "Transformer", "CAG-comp", "CAG"),
  PPP_mean = c(
    mean(loglik_lstm_values),
    mean(loglik_actionlstm_values),
    mean(loglik_rnng_values),
    mean(loglik_transformer_values),
    mean(loglik_plm_values),
    mean(loglik_cag_values)
  ),
  PPP_sd = c(
    sd(loglik_lstm_values),
    sd(loglik_actionlstm_values),
    sd(loglik_rnng_values),
    sd(loglik_transformer_values),
    sd(loglik_plm_values),
    sd(loglik_cag_values)
  )
)

plot_data$Model <- factor(plot_data$Model, 
                          levels = c("LSTM", "RNNG-comp", "RNNG", "Transformer", "CAG-comp", "CAG"))

colors <- c("LSTM" = "#2E7D32",
            "RNNG-comp" = "#66BB6A",
            "RNNG" = "#A5D6A7",
            "Transformer" = "#FF8F00",
            "CAG-comp" = "#FFD54F",
            "CAG" = "#FFEB3B")

y_max <- max(plot_data$PPP_mean + plot_data$PPP_sd) * 1.1
y_min <- 0

p <- ggplot(plot_data, aes(x = Model, y = PPP_mean, fill = Model)) +
  geom_bar(stat = "identity", width = 0.7, color = "black", size = 0.3) +
  geom_errorbar(aes(ymin = PPP_mean - PPP_sd, ymax = PPP_mean + PPP_sd),
                width = 0.2, size = 0.5) +
  scale_fill_manual(values = colors) +
  scale_y_continuous(limits = c(y_min, y_max)) +
  labs(x = "", y = "PPP") +
  theme_minimal() +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "grey80", linetype = "dashed"),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_text(size = 10),
    axis.title.y = element_text(size = 12),
    legend.position = "right",
    legend.title = element_blank(),
    legend.text = element_text(size = 10),
    legend.key.size = unit(0.8, "cm"),
    plot.margin = margin(10, 10, 10, 10)
  ) +
  geom_hline(yintercept = 0, color = "black", size = 0.5) +
  geom_segment(aes(x = 0.4, xend = 0.4, y = 0, yend = y_max), color = "black", size = 0.5)

ggsave("results/ppp_comparison.png", plot = p, width = 6.5, height = 3.0, dpi = 300, device = "png")

# recurrence vs. attention
## cag v.s. rnng
GD_lmer_cag_rnng_ave = lmer(
  log(GD) ~ surp_cag_ave + prev_surp_cag_ave + surp_rnng_ave + prev_surp_rnng_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)

summary(GD_lmer_cag_rnng_ave)
anova(GD_lmer_cag_ave, GD_lmer_cag_rnng_ave)
anova(GD_lmer_rnng_ave, GD_lmer_cag_rnng_ave)

## plm v.s. adtionlstm
GD_lmer_plm_actionlstm_ave = lmer(
  log(GD) ~ surp_plm_ave + surp_actionlstm_ave + prev_surp_plm_ave + prev_surp_actionlstm_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_plm_actionlstm_ave)
anova(GD_lmer_plm_ave, GD_lmer_plm_actionlstm_ave)
anova(GD_lmer_actionlstm_ave, GD_lmer_plm_actionlstm_ave)

## transformer v.s. lstm
GD_lmer_transformer_lstm_ave = lmer(
  log(GD) ~ surp_transformer_ave + surp_lstm_ave + prev_surp_transformer_ave + prev_surp_lstm_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_transformer_lstm_ave)
anova(GD_lmer_transformer_ave, GD_lmer_transformer_lstm_ave)
anova(GD_lmer_lstm_ave, GD_lmer_transformer_lstm_ave)

# syntactic vs. sequential (attention)
## cag v.s. plm
GD_lmer_cag_plm_ave = lmer(
  log(GD) ~ surp_cag_ave + surp_plm_ave + prev_surp_cag_ave + prev_surp_plm_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_cag_plm_ave)
anova(GD_lmer_cag_ave, GD_lmer_cag_plm_ave)
anova(GD_lmer_plm_ave, GD_lmer_cag_plm_ave)


## cag v.s transformer
GD_lmer_cag_transformer_ave = lmer(
  log(GD) ~ surp_cag_ave + surp_transformer_ave + prev_surp_cag_ave + prev_surp_transformer_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_cag_transformer_ave)
anova(GD_lmer_cag_ave, GD_lmer_cag_transformer_ave)
anova(GD_lmer_transformer_ave, GD_lmer_cag_transformer_ave)

## plm v.s. transformer
GD_lmer_plm_transformer_ave = lmer(
  log(GD) ~ surp_plm_ave + surp_transformer_ave + prev_surp_plm_ave + prev_surp_transformer_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_plm_transformer_ave)
anova(GD_lmer_plm_ave, GD_lmer_plm_transformer_ave)
anova(GD_lmer_transformer_ave, GD_lmer_plm_transformer_ave)

# syntactic vs. sequential (recurrence)
## rnng v.s. actionlstm
GD_lmer_rnng_actionlstm_ave = lmer(
  log(GD) ~ surp_rnng_ave + surp_actionlstm_ave + prev_surp_rnng_ave + prev_surp_actionlstm_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_rnng_actionlstm_ave)
anova(GD_lmer_rnng_ave, GD_lmer_rnng_actionlstm_ave)
anova(GD_lmer_actionlstm_ave, GD_lmer_rnng_actionlstm_ave)


# rnng v.s lstm
GD_lmer_rnng_lstm_ave = lmer(
  log(GD) ~ surp_rnng_ave + surp_lstm_ave + prev_surp_rnng_ave + prev_surp_lstm_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_rnng_lstm_ave)
anova(GD_lmer_rnng_ave, GD_lmer_rnng_lstm_ave)
anova(GD_lmer_lstm_ave, GD_lmer_rnng_lstm_ave)


# actionlstm v.s. lstm
GD_lmer_actionlstm_lstm_ave = lmer(
  log(GD) ~ surp_actionlstm_ave + surp_lstm_ave + prev_surp_actionlstm_ave + prev_surp_lstm_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWhole,
  REML = FALSE
)
summary(GD_lmer_actionlstm_lstm_ave)
anova(GD_lmer_actionlstm_ave, GD_lmer_actionlstm_lstm_ave)
anova(GD_lmer_lstm_ave, GD_lmer_actionlstm_lstm_ave)

# Analysis for max_word_id above average
dataAboveAverage = data[data$max_word_id >= mean(data$max_word_id),]

# hist (exploratory; skipped in non-interactive runs)
if (interactive()) {
  hist(log(dataAboveAverage$GD))
  dev.off()
  kurtosis(log(dataAboveAverage$GD))

  hist(dataAboveAverage$length)
  hist(dataAboveAverage$prev_length)

  hist(dataAboveAverage$freq)
  hist(dataAboveAverage$prev_freq)

  hist(dataAboveAverage$screen_N)
  hist(dataAboveAverage$line_N)
  hist(dataAboveAverage$segment_N)

  hist(dataAboveAverage$sent_id)
  hist(dataAboveAverage$word_id)
}

# fitting baseline
control = lmerControl(optCtrl=list(maxfun=100000))
GD_lmer_above = lmer(
  log(GD) ~ length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataAboveAverage,
  REML = FALSE
)
summary(GD_lmer_above)
coefplot(GD_lmer_above)

# trimming outlier
dataTrimWholeAbove = dataAboveAverage[scale(resid(GD_lmer_above)) < 3.0, ]
nrow(dataAboveAverage) - nrow(dataTrimWholeAbove)
GD_lmer_above = lmer(
  log(GD) ~ length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeAbove,
  REML = FALSE
)
summary(GD_lmer_above)
coefplot(GD_lmer_above)

# fitting
## add cag surp
GD_lmer_cag_seed3435_above = lmer(
  log(GD) ~ surp_cag_seed3435 + prev_surp_cag_seed3435 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeAbove,
  REML = FALSE
)
summary(GD_lmer_cag_seed3435_above)

GD_lmer_cag_seed3436_above = lmer(
  log(GD) ~ surp_cag_seed3436 + prev_surp_cag_seed3436 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeAbove,
  REML = FALSE
)
summary(GD_lmer_cag_seed3436_above)

GD_lmer_cag_seed3437_above = lmer(
  log(GD) ~ surp_cag_seed3437 + prev_surp_cag_seed3437 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeAbove,
  REML = FALSE
)
summary(GD_lmer_cag_seed3437_above)

GD_lmer_cag_ave_above = lmer(
  log(GD) ~ surp_cag_ave + prev_surp_cag_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeAbove,
  REML = FALSE
)
summary(GD_lmer_cag_ave_above)


## add rnng surp
GD_lmer_rnng_seed3435_above = lmer(
  log(GD) ~ surp_rnng_seed3435 + prev_surp_rnng_seed3435 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeAbove,
  REML = FALSE
)
summary(GD_lmer_rnng_seed3435_above)

GD_lmer_rnng_seed3436_above = lmer(
  log(GD) ~ surp_rnng_seed3436 + prev_surp_rnng_seed3436 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeAbove,
  REML = FALSE
)
summary(GD_lmer_rnng_seed3436_above)

GD_lmer_rnng_seed3437_above = lmer(
  log(GD) ~ surp_rnng_seed3437 + prev_surp_rnng_seed3437 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeAbove,
  REML = FALSE
)
summary(GD_lmer_rnng_seed3437_above)

GD_lmer_rnng_ave_above = lmer(
  log(GD) ~ surp_rnng_ave + prev_surp_rnng_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeAbove,
  REML = FALSE
)
summary(GD_lmer_rnng_seed3437_above)


# anova
## anova cag
anova(GD_lmer_above, GD_lmer_cag_seed3435_above)
loglik_cag_3435_above = logLik(GD_lmer_cag_seed3435_above) - logLik(GD_lmer_above)

anova(GD_lmer_above, GD_lmer_cag_seed3436_above)
loglik_cag_3436_above = logLik(GD_lmer_cag_seed3436_above) - logLik(GD_lmer_above) 

anova(GD_lmer_above, GD_lmer_cag_seed3437_above)
loglik_cag_3437_above = logLik(GD_lmer_cag_seed3437_above) - logLik(GD_lmer_above) 

loglik_cag_above = c(loglik_cag_3435_above,loglik_cag_3436_above,loglik_cag_3437_above)
print(loglik_cag_above)
mean(loglik_cag_above)
sd(loglik_cag_above)

anova(GD_lmer_above, GD_lmer_cag_ave_above)
loglik_cag_ave_above = logLik(GD_lmer_cag_ave_above) - logLik(GD_lmer_above)
print(loglik_cag_ave_above)


## anova rnng
anova(GD_lmer_above, GD_lmer_rnng_seed3435_above)
loglik_rnng_3435_above = logLik(GD_lmer_rnng_seed3435_above) - logLik(GD_lmer_above) 

anova(GD_lmer_above, GD_lmer_rnng_seed3436_above)
loglik_rnng_3436_above = logLik(GD_lmer_rnng_seed3436_above) - logLik(GD_lmer_above) 

anova(GD_lmer_above, GD_lmer_rnng_seed3437_above)
loglik_rnng_3437_above = logLik(GD_lmer_rnng_seed3437_above) - logLik(GD_lmer_above)

loglik_rnng_above = c(loglik_rnng_3435_above,loglik_rnng_3436_above,loglik_rnng_3437_above)
print(loglik_rnng_above)
mean(loglik_rnng_above)
sd(loglik_rnng_above)

anova(GD_lmer_above, GD_lmer_rnng_ave_above)
loglik_rnng_ave_above = logLik(GD_lmer_rnng_ave_above) - logLik(GD_lmer_above) 
print(loglik_rnng_ave_above)


# recurrence vs. attention
## cag v.s. rnng
GD_lmer_cag_rnng_ave_above = lmer(
  log(GD) ~ surp_cag_ave + prev_surp_cag_ave + surp_rnng_ave + prev_surp_rnng_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeAbove,
  REML = FALSE
)
summary(GD_lmer_cag_rnng_ave_above)
anova(GD_lmer_cag_ave_above, GD_lmer_cag_rnng_ave_above)
anova(GD_lmer_rnng_ave_above, GD_lmer_cag_rnng_ave_above)


# Analysis for max_word_id below average
dataBelowAverage = data[data$max_word_id < mean(data$max_word_id),]

# hist (exploratory; skipped in non-interactive runs)
if (interactive()) {
  hist(log(dataBelowAverage$GD))
  dev.off()
  kurtosis(log(dataBelowAverage$GD))

  hist(dataBelowAverage$length)
  hist(dataBelowAverage$prev_length)

  hist(dataBelowAverage$freq)
  hist(dataBelowAverage$prev_freq)

  hist(dataBelowAverage$screen_N)
  hist(dataBelowAverage$line_N)
  hist(dataBelowAverage$segment_N)

  hist(dataBelowAverage$sent_id)
  hist(dataBelowAverage$word_id)
}

# fitting baseline
control = lmerControl(optCtrl=list(maxfun=100000))
GD_lmer_below = lmer(
  log(GD) ~ length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataBelowAverage,
  REML = FALSE
)
summary(GD_lmer_below)
coefplot(GD_lmer_below)

# trimming outlier
dataTrimWholeBelow = dataBelowAverage[scale(resid(GD_lmer_below)) < 3.0, ]
nrow(dataBelowAverage) - nrow(dataTrimWholeBelow)
GD_lmer_below = lmer(
  log(GD) ~ length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeBelow,
  REML = FALSE
)
summary(GD_lmer_below)
coefplot(GD_lmer_below)

# fitting
## add cag surp
GD_lmer_cag_seed3435_below = lmer(
  log(GD) ~ surp_cag_seed3435 + prev_surp_cag_seed3435 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeBelow,
  REML = FALSE
)
summary(GD_lmer_cag_seed3435_below)

GD_lmer_cag_seed3436_below = lmer(
  log(GD) ~ surp_cag_seed3436 + prev_surp_cag_seed3436 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeBelow,
  REML = FALSE
)
summary(GD_lmer_cag_seed3436_below)

GD_lmer_cag_seed3437_below = lmer(
  log(GD) ~ surp_cag_seed3437 + prev_surp_cag_seed3437 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeBelow,
  REML = FALSE
)
summary(GD_lmer_cag_seed3437_below)

GD_lmer_cag_ave_below = lmer(
  log(GD) ~ surp_cag_ave + prev_surp_cag_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeBelow,
  REML = FALSE
)
summary(GD_lmer_cag_ave_below)


## add rnng surp
GD_lmer_rnng_seed3435_below = lmer(
  log(GD) ~ surp_rnng_seed3435 + prev_surp_rnng_seed3435 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeBelow,
  REML = FALSE
)
summary(GD_lmer_rnng_seed3435_below)

GD_lmer_rnng_seed3436_below = lmer(
  log(GD) ~ surp_rnng_seed3436 + prev_surp_rnng_seed3436 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeBelow,
  REML = FALSE
)
summary(GD_lmer_rnng_seed3436_below)

GD_lmer_rnng_seed3437_below = lmer(
  log(GD) ~ surp_rnng_seed3437 + prev_surp_rnng_seed3437 + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeBelow,
  REML = FALSE
)
summary(GD_lmer_rnng_seed3437_below)

GD_lmer_rnng_ave_below = lmer(
  log(GD) ~ surp_rnng_ave + prev_surp_rnng_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeBelow,
  REML = FALSE
)
summary(GD_lmer_rnng_seed3437_below)


# anova
## anova cag
anova(GD_lmer_below, GD_lmer_cag_seed3435_below)
loglik_cag_3435_below = logLik(GD_lmer_cag_seed3435_below) - logLik(GD_lmer_below)

anova(GD_lmer_below, GD_lmer_cag_seed3436_below)
loglik_cag_3436_below = logLik(GD_lmer_cag_seed3436_below) - logLik(GD_lmer_below) 

anova(GD_lmer_below, GD_lmer_cag_seed3437_below)
loglik_cag_3437_below = logLik(GD_lmer_cag_seed3437_below) - logLik(GD_lmer_below) 

loglik_cag_below = c(loglik_cag_3435_below,loglik_cag_3436_below,loglik_cag_3437_below)
print(loglik_cag_below)
mean(loglik_cag_below)
sd(loglik_cag_below)

anova(GD_lmer_below, GD_lmer_cag_ave_below)
loglik_cag_ave_below = logLik(GD_lmer_cag_ave_below) - logLik(GD_lmer_below)
print(loglik_cag_ave_below)


## anova rnng
anova(GD_lmer_below, GD_lmer_rnng_seed3435_below)
loglik_rnng_3435_below = logLik(GD_lmer_rnng_seed3435_below) - logLik(GD_lmer_below) 

anova(GD_lmer_below, GD_lmer_rnng_seed3436_below)
loglik_rnng_3436_below = logLik(GD_lmer_rnng_seed3436_below) - logLik(GD_lmer_below) 

anova(GD_lmer_below, GD_lmer_rnng_seed3437_below)
loglik_rnng_3437_below = logLik(GD_lmer_rnng_seed3437_below) - logLik(GD_lmer_below)

loglik_rnng_below = c(loglik_rnng_3435_below,loglik_rnng_3436_below,loglik_rnng_3437_below)
print(loglik_rnng_below)
mean(loglik_rnng_below)
sd(loglik_rnng_below)

anova(GD_lmer_below, GD_lmer_rnng_ave_below)
loglik_rnng_ave_below = logLik(GD_lmer_rnng_ave_below) - logLik(GD_lmer_below) 
print(loglik_rnng_ave_below)


# recurrence vs. attention
## cag v.s. rnng
GD_lmer_cag_rnng_ave_below = lmer(
  log(GD) ~ surp_cag_ave + prev_surp_cag_ave + surp_rnng_ave + prev_surp_rnng_ave + length + prev_length + freq + prev_freq + sent_id + word_id + (1 |subj),
  dataTrimWholeBelow,
  REML = FALSE
)
summary(GD_lmer_cag_rnng_ave_below)
anova(GD_lmer_cag_ave_below, GD_lmer_cag_rnng_ave_below)
anova(GD_lmer_rnng_ave_below, GD_lmer_cag_rnng_ave_below)

# plot
length_n_datapoints_below <- nrow(dataTrimWholeBelow)
length_n_datapoints_above <- nrow(dataTrimWholeAbove)

length_loglik_cag_below_values = c(loglik_cag_3435_below, loglik_cag_3436_below, loglik_cag_3437_below) / length_n_datapoints_below
length_loglik_rnng_below_values = c(loglik_rnng_3435_below, loglik_rnng_3436_below, loglik_rnng_3437_below) / length_n_datapoints_below

length_loglik_cag_above_values = c(loglik_cag_3435_above, loglik_cag_3436_above, loglik_cag_3437_above) / length_n_datapoints_above
length_loglik_rnng_above_values = c(loglik_rnng_3435_above, loglik_rnng_3436_above, loglik_rnng_3437_above) / length_n_datapoints_above

length_plot_data_below <- data.frame(
  Model = c("RNNG", "CAG"),
  PPP_mean = c(
    mean(length_loglik_rnng_below_values),
    mean(length_loglik_cag_below_values)
  ),
  PPP_sd = c(
    sd(length_loglik_rnng_below_values),
    sd(length_loglik_cag_below_values)
  ),
  Length = "Short\nsentences"
)

length_plot_data_above <- data.frame(
  Model = c("RNNG", "CAG"),
  PPP_mean = c(
    mean(length_loglik_rnng_above_values),
    mean(length_loglik_cag_above_values)
  ),
  PPP_sd = c(
    sd(length_loglik_rnng_above_values),
    sd(length_loglik_cag_above_values)
  ),
  Length = "Long\nsentences"
)

length_plot_data <- rbind(length_plot_data_below, length_plot_data_above)
length_plot_data$Model <- factor(length_plot_data$Model, levels = c("RNNG", "CAG"))
length_plot_data$Length <- factor(length_plot_data$Length, levels = c("Short\nsentences", "Long\nsentences"))

length_colors <- c("RNNG" = "#A5D6A7", "CAG" = "#FFEB3B")

length_y_max <- max(length_plot_data$PPP_mean + length_plot_data$PPP_sd) * 1.1
length_y_min <- 0

length_p <- ggplot(length_plot_data, aes(x = Model, y = PPP_mean, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6, color = "black", size = 0.3) +
  geom_errorbar(aes(ymin = PPP_mean - PPP_sd, ymax = PPP_mean + PPP_sd),
                width = 0.15, size = 0.5) +
  scale_fill_manual(values = length_colors) +
  scale_y_continuous(limits = c(length_y_min, length_y_max)) +
  labs(x = "", y = "PPP") +
  theme_minimal() +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "grey80", linetype = "dashed"),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_text(size = 8),
    axis.title.y = element_text(size = 9),
    legend.position = "right",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.6, "cm"),
    strip.text = element_text(size = 8, margin = margin(b = 6)),
    plot.margin = margin(12, 8, 6, 8)
  ) +
  geom_hline(yintercept = 0, color = "black", size = 0.5) +
  geom_segment(aes(x = 0.4, xend = 0.4, y = 0, yend = length_y_max), color = "black", size = 0.5) +
  facet_wrap(~ Length, ncol = 2, scales = "free_y")

ggsave("results/ppp_length_comparison.png", plot = length_p, width = 3.3, height = 2.8, dpi = 300, device = "png")

