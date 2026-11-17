# Coefficient of determination (R^2) of the regression models.
#
# Supplementary analysis reported in a footnote to Chapter 3 of
#   Ryo Yoshida (2027). The Hybrid Approach to Computational Psycholinguistics in the Era
#   of Transformers. Doctoral dissertation, The University of Tokyo.
# It is not part of the CoNLL 2025 paper.
#
# For the baseline model and for each language model and seed, this script fits the same
# mixed effects models as modeling_gd.R (on the same data, after the same exclusion and
# outlier removal) and reports the marginal and conditional R^2 of Nakagawa and
# Schielzeth (2013), the increases in marginal and conditional R^2 over the baseline
# (Delta R^2), the variance of the random intercepts, and the increase in log-likelihood
# per data point (Delta LogLik).
#
# Input:  ../data/gd_analysis.csv.gz (the bundled analysis table)
# Output: results/r_squared.csv, and a summary printed to the standard output.
#
# Run with this r-workspace directory as the working directory:
#   Rscript r_squared.R

library(lme4)

is_true = function(x) toupper(as.character(x)) == 'TRUE'

MODELS = c(
  LSTM = 'lstm',
  'RNNG-comp' = 'actionlstm',
  RNNG = 'rnng',
  Transformer = 'transformer',
  'CAG-comp' = 'plm',
  CAG = 'cag'
)
SEEDS = c('3435', '3436', '3437')

# data exclusion of modeling_gd.R
data = read.csv('../data/gd_analysis.csv.gz')
data = data[data$freq > 0 & data$prev_freq > 0, ]
data = data[!is_true(data$is_sent_first) & !is_true(data$is_sent_last), ]

data$length = scale(data$length)
data$prev_length = scale(data$prev_length)
data$freq = scale(log2(data$freq))
data$prev_freq = scale(log2(data$prev_freq))
data$sent_id = scale(data$sent_id)
data$word_id = scale(data$word_id)
data$subj = as.factor(data$subj)
for (m in MODELS) for (s in SEEDS) {
  for (v in c(paste0('surp_', m, '_seed', s), paste0('prev_surp_', m, '_seed', s))) {
    data[[v]] = scale(data[[v]])
  }
}

baseline_formula = 'log(GD) ~ length + prev_length + freq + prev_freq + sent_id + word_id + (1 | subj)'
fit = function(formula, data) lmer(as.formula(formula), data, REML = FALSE)

# outlier removal of modeling_gd.R
baseline = fit(baseline_formula, data)
data = data[scale(resid(baseline)) < 3.0, ]
baseline = fit(baseline_formula, data)
n = nrow(data)

# marginal and conditional R^2 (Nakagawa and Schielzeth, 2013) of a random-intercept model
r_squared = function(model) {
  var_fixed = var(as.vector(model.matrix(model) %*% fixef(model)))
  var_random = sum(sapply(VarCorr(model), function(v) v[1]))
  var_residual = sigma(model)^2
  total = var_fixed + var_random + var_residual
  c(marginal = var_fixed / total, conditional = (var_fixed + var_random) / total,
    var_random = var_random)
}

r2_baseline = r_squared(baseline)
loglik_baseline = as.numeric(logLik(baseline))

rows = data.frame(model = 'Baseline', seed = NA, marginal = r2_baseline[['marginal']],
                  conditional = r2_baseline[['conditional']], var_random = r2_baseline[['var_random']],
                  delta_r2 = NA, delta_r2_conditional = NA, delta_loglik = NA)
for (label in names(MODELS)) for (s in SEEDS) {
  surp = paste0('surp_', MODELS[[label]], '_seed', s)
  formula = sub('~', paste0('~ ', surp, ' + prev_', surp, ' +'), baseline_formula, fixed = TRUE)
  model = fit(formula, data)
  r2 = r_squared(model)
  rows = rbind(rows, data.frame(
    model = label, seed = s, marginal = r2[['marginal']], conditional = r2[['conditional']],
    var_random = r2[['var_random']],
    delta_r2 = r2[['marginal']] - r2_baseline[['marginal']],
    delta_r2_conditional = r2[['conditional']] - r2_baseline[['conditional']],
    delta_loglik = (as.numeric(logLik(model)) - loglik_baseline) / n
  ))
}

dir.create('results', showWarnings = FALSE)
write.csv(rows, 'results/r_squared.csv', row.names = FALSE)

cat('Data points:', n, '\n\n')
cat(sprintf('Baseline: marginal R^2 = %.4f, conditional R^2 = %.4f, random-intercept variance = %.5f\n\n',
            r2_baseline[['marginal']], r2_baseline[['conditional']], r2_baseline[['var_random']]))
cat('Mean over seeds (models ordered by Delta LogLik)\n')
summary = aggregate(cbind(marginal, conditional, var_random, delta_r2, delta_r2_conditional, delta_loglik) ~ model,
                    rows[rows$model != 'Baseline', ], mean)
summary = summary[order(summary$delta_loglik, decreasing = TRUE), ]
summary$rank_delta_r2 = rank(-summary$delta_r2)
print(format(summary, digits = 4), row.names = FALSE)
