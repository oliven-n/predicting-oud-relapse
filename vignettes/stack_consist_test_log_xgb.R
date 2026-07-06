library(stacks)
library(tidymodels)
library(xgboost)
library(dplyr)
library(ggplot2)
library(vip)
library(conflicted)
conflicts_prefer(dplyr::filter)
conflicts_prefer(purrr::discard)

# Source logistic_wopv to get lr_recipe_wopv, train_data_wopv, test_data_wopv,
# data_split_wopv, lr_workflow_wopv. Guard prevents double-source.
if (!exists("train_data_wopv")) {
  source(here::here("vignettes/logistic_regression_without_problem_vars.R"))
}
source(here::here("vignettes/metrics_helpers.R"))

# Load XGBoost best hyperparameters from the standalone tuning cache.
# We do NOT re-tune — xgboost_favorite carries the already-selected
# learn_rate / tree_depth / loss_reduction / mtry / sample_size.
xgboost_tune_loaded <- readRDS(here::here("vignettes/xgboost_tune.rds"))
xgboost_favorite    <- select_best(xgboost_tune_loaded, metric = "roc_auc")

# Workflow ----------------
# stacks requires a workflow for each level-1 model. Both use lr_recipe_wopv so
# the feature set (wopv exclusions applied) is consistent across the ensemble.
# The level-2 meta-learner is LASSO logistic regression, fitted automatically
# by blend_predictions() — no separate workflow object is needed for it.

# Level-1: logistic (non-penalized). Reuse lr_workflow_wopv from sourced file.
stack_lr_workflow <- lr_workflow_wopv

# Level-1: XGBoost at the already-tuned best hyperparameters.
# Hard-code the selected values directly into the spec (no tune() calls) so
# fit_resamples() runs a single fixed model rather than a grid search.
stack_xgb_spec <-
  boost_tree(
    trees          = 5000,
    learn_rate     = xgboost_favorite$learn_rate,
    tree_depth     = xgboost_favorite$tree_depth,
    loss_reduction = xgboost_favorite$loss_reduction,
    mtry           = xgboost_favorite$mtry,
    sample_size    = xgboost_favorite$sample_size,
    stop_iter      = 50
  ) |>
  set_engine("xgboost") |>
  set_mode("classification")

stack_xgb_workflow <- workflow() |>
  add_recipe(lr_recipe_wopv) |>
  add_model(stack_xgb_spec)

# Level-1 Resampling --------
# stacks requires both level-1 models to produce out-of-fold (OOF) predictions
# via fit_resamples() run with control_stack_resamples(), which internally sets
# save_pred = TRUE and save_workflow = TRUE.
#
# WHY NOT REUSE EXISTING .rds CACHES:
#   xgboost_tune.rds: was run with plain control_grid() — no save_pred = TRUE —
#   so per-observation OOF predictions were never stored; add_candidates() would error.
#   logistic_regression_without_problem_vars.R: never called fit_resamples() at all,
#   only fit() on the full training set. Both level-1 models must be re-run here.
#
# Both models share the_folds_stack (same seed → same 5 splits) so OOF predictions
# are fold-aligned when blend_predictions() trains the meta-learner.

set.seed(12345)
the_folds_stack <- vfold_cv(train_data_wopv, v = 5, strata = outcome)

stack_lr_resamples_cache <- here::here("vignettes/stack_lr_resamples.rds")
if (file.exists(stack_lr_resamples_cache)) {
  stack_lr_resamples <- readRDS(stack_lr_resamples_cache)
} else {
  stack_lr_resamples <- fit_resamples(
    stack_lr_workflow,
    resamples = the_folds_stack,
    control   = control_stack_resamples()
  )
  saveRDS(stack_lr_resamples, stack_lr_resamples_cache)
}

stack_xgb_resamples_cache <- here::here("vignettes/stack_xgb_resamples.rds")
if (file.exists(stack_xgb_resamples_cache)) {
  stack_xgb_resamples <- readRDS(stack_xgb_resamples_cache)
} else {
  # XGBoost at fixed (already-tuned) hyperparameters — 5-fold fit, ~minutes.
  stack_xgb_resamples <- fit_resamples(
    stack_xgb_workflow,
    resamples = the_folds_stack,
    control   = control_stack_resamples()
  )
  saveRDS(stack_xgb_resamples, stack_xgb_resamples_cache)
}

# Model Fit --------
# blend_predictions() fits a LASSO (mixture = 1) logistic meta-learner on the
# OOF predictions from both level-1 models — this IS the level-2 logistic regression.
# It selects which members survive regularization and at what blend weight.
# fit_members() re-fits the surviving level-1 members on ALL of train_data_wopv,
# then bundles them with the meta-learner into a single predict()-able model_stack.

stack_blend_cache <- here::here("vignettes/stack_blend.rds")
stack_fit_cache   <- here::here("vignettes/stack_fit.rds")

if (file.exists(stack_fit_cache) && file.exists(stack_blend_cache)) {
  stack_blend <- readRDS(stack_blend_cache)
  stack_fit   <- readRDS(stack_fit_cache)
} else {
  stack_data <- stacks() |>
    add_candidates(stack_lr_resamples) |>
    add_candidates(stack_xgb_resamples)

  stack_blend <- blend_predictions(stack_data)
  stack_fit   <- fit_members(stack_blend)
  saveRDS(stack_blend, stack_blend_cache)
  saveRDS(stack_fit,   stack_fit_cache)
}

# Inspect member selection: which level-1 models survived LASSO and at what weight.
autoplot(stack_blend)                    # OOF AUC across the penalty path
autoplot(stack_blend, type = "members")  # number of members retained per penalty
autoplot(stack_blend, type = "weights")  # blending coefficients at selected penalty

# CV Metrics --------
# Divergence from xgboost.R: stacks' blend step evaluates the meta-learner across
# a LASSO penalty path using the OOF predictions as input. The AUC reported here
# is the OOF roc_auc at the best-performing penalty — not a separate outer 5-fold
# CV of the full stacked pipeline. mean / std_err / n match other *_cv_metrics
# tables so analysis.qmd can row-bind it.
# collect_metrics() has no method for <linear_stack>; $metrics is the same tibble
# autoplot(stack_blend) uses internally (columns: penalty, .metric, mean, n, std_err).
stack_cv_metrics <- stack_blend$metrics |>
  dplyr::filter(.metric == "roc_auc") |>
  dplyr::arrange(desc(mean)) |>
  dplyr::slice(1) |>
  dplyr::select(.metric, mean, std_err, n)
stack_cv_metrics

# Train Metrics --------
# Divergence: stack_fit was trained on ALL of train_data_wopv; relapse_pred_stack
# is in-sample prediction — intentionally optimistic, equivalent to Balise et al.'s
# "Full Training Dataset" column.
stack_train_pred_cache <- here::here("vignettes/stack_train_pred.rds")
if (file.exists(stack_train_pred_cache)) {
  relapse_pred_stack <- readRDS(stack_train_pred_cache)
} else {
  proba_train <- predict(stack_fit, new_data = train_data_wopv, type = "prob")
  relapse_pred_stack <- tibble(
    .pred_0 = proba_train$.pred_0,
    .pred_1 = proba_train$.pred_1,
    outcome = train_data_wopv$outcome
  )
  saveRDS(relapse_pred_stack, stack_train_pred_cache)
}

stack_cut <- youden_cutoff(relapse_pred_stack)
stack_cut

stack_train_metrics <- bind_rows(
  roc_auc(relapse_pred_stack, truth = outcome, .pred_1, event_level = "second"),
  sens_spec_at(relapse_pred_stack, stack_cut)
) |> select(.metric, .estimate)
stack_train_metrics


# Review Fit on Training Data --------
relapse_pred_stack |>
  roc_curve(truth = outcome, .pred_1, event_level = "second") |>
  autoplot() +
  labs(
    title    = "Stack (LR + XGB → LR): training ROC curve",
    subtitle = "Relapse as the positive class; logistic + XGBoost level-1, LASSO meta-learner",
    x = "1 - Specificity", y = "Sensitivity"
  )

relapse_pred_stack |>
  roc_curve(truth = outcome, .pred_1, event_level = "second")

# Test Metrics --------
# No last_fit object → build the method/.metric/.estimate tibble by hand to match
# the helper's schema so analysis.qmd still row-binds it.
# Structure verified against knn_test_metrics: same column names/order/types
# and same .metric order: accuracy, roc_auc, sens, spec.
# Youden-J cutoff chosen on train_data_wopv (stack_cut), applied here.
stack_test_pred_cache <- here::here("vignettes/stack_test_pred.rds")
if (file.exists(stack_test_pred_cache)) {
  relapse_pred_stack_test <- readRDS(stack_test_pred_cache)
} else {
  proba_test <- predict(stack_fit, new_data = test_data_wopv, type = "prob")
  relapse_pred_stack_test <- tibble(
    .pred_0 = proba_test$.pred_0,
    .pred_1 = proba_test$.pred_1,
    outcome = test_data_wopv$outcome
  )
  saveRDS(relapse_pred_stack_test, stack_test_pred_cache)
}

# accuracy at the tidymodels default 0.5 threshold (matches collect_metrics()),
# then roc_auc, then sens/spec at the TRAIN-chosen cutoff — same order
# test_metrics_from_lastfit() emits (collect_metrics' accuracy+roc_auc, then ss).
acc_at_half <- relapse_pred_stack_test |>
  mutate(.pred_class = factor(if_else(.pred_1 >= 0.5, "1", "0"),
                              levels = c("0", "1"))) |>
  accuracy(truth = outcome, estimate = .pred_class)

stack_test_metrics <- bind_rows(
  acc_at_half,
  roc_auc(relapse_pred_stack_test, truth = outcome, .pred_1, event_level = "second"),
  sens_spec_at(relapse_pred_stack_test, stack_cut)
) |>
  transmute(method = "Stack (LR + XGB → LR)", .metric, .estimate)

stack_test_metrics


# Review Fit on Test Data --------
relapse_pred_stack_test |>
  roc_curve(truth = outcome, .pred_1, event_level = "second") |>
  autoplot() +
  labs(
    title    = "Stack (LR + XGB → LR): test ROC curve",
    subtitle = "Held-out 25% test split; relapse as the positive class",
    x = "1 - Specificity", y = "Sensitivity"
  )

# Prediction Probabilities — density of predicted P(relapse) by true outcome
relapse_pred_stack_test |>
  ggplot2::ggplot(ggplot2::aes(x = .pred_1, fill = outcome)) +
  ggplot2::geom_density(alpha = 0.5) +
  ggplot2::labs(
    title = "Stack: True Outcome vs Prediction Probability",
    x     = "Probability of Relapse",
    y     = "Density"
  ) +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.title = element_blank())

# Experiment --------
# Does adding logistic regression as a level-1 candidate change the stacked
# ensemble's test ROC AUC relative to XGBoost alone?
#
# Method: vary the random seed (controls train/test split + CV fold assignment)
# across 20 values. For each seed, build two meta-learners on a fresh 75/25 split:
#   Condition A: LR + XGB OOF predictions → LASSO  (via stacks)
#   Condition B: XGB OOF predictions only → LASSO   (manual cv.glmnet — stacks
#                                                     requires ≥2 candidates)
# Both conditions share the same seed, same splits, and same XGB resamples,
# so a paired t-test isolates the effect of adding LR.
#
# Expected result: NOT significantly different (p > 0.05). XGBoost already
# captures any linear signal logistic finds; LASSO is expected to zero out or
# near-zero the logistic member. A significant result would be concerning —
# it would suggest logistic is destabilizing rather than improving the ensemble.

stack_experiment_cache <- here::here("vignettes/stack_experiment_results.rds")

if (file.exists(stack_experiment_cache)) {
  sens_results <- readRDS(stack_experiment_cache)
} else {
  seeds <- c(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L,
             111L, 222L, 333L, 444L, 555L, 666L, 777L, 888L, 999L, 12345L)

  n_seeds       <- length(seeds)
  n_folds       <- 5L
  fits_per_seed <- n_folds * 2L
  total_fits    <- n_seeds * fits_per_seed
  fit_count     <- 0L

  run_one_seed <- function(seed) {
    seed_cache <- here::here(sprintf("vignettes/stack_sens_seed_%d.rds", seed))
    if (file.exists(seed_cache)) {
      fit_count <<- fit_count + fits_per_seed
      cat(sprintf("[%d/%d fits] seed %d loaded from cache\n", fit_count, total_fits, seed))
      return(readRDS(seed_cache))
    }

    set.seed(seed)
    split <- initial_split(analysis_tibble_wopv, prop = 0.75, strata = outcome)
    train <- training(split)
    test  <- testing(split)
    folds <- vfold_cv(train, v = 5, strata = outcome)

    xgb_wf_exp <- workflow() |>
      add_recipe(lr_recipe_wopv) |>
      add_model(stack_xgb_spec)

    lr_res_exp <- fit_resamples(lr_workflow_wopv, resamples = folds,
                                control = control_stack_resamples())
    fit_count <<- fit_count + n_folds
    cat(sprintf("[%d/%d fits] seed %d — LR resamples done\n", fit_count, total_fits, seed))

    xgb_res_exp <- fit_resamples(xgb_wf_exp, resamples = folds,
                                 control = control_stack_resamples())
    fit_count <<- fit_count + n_folds
    cat(sprintf("[%d/%d fits] seed %d — XGB resamples done\n", fit_count, total_fits, seed))

    get_test_auc <- function(stack_data_obj) {
      preds <- predict(fit_members(blend_predictions(stack_data_obj)),
                       new_data = test, type = "prob") |>
        bind_cols(test |> dplyr::select(outcome))
      roc_auc(preds, truth = outcome, .pred_1, event_level = "second")$.estimate
    }

    auc_a <- get_test_auc(stacks() |> add_candidates(lr_res_exp) |> add_candidates(xgb_res_exp))

    # Condition B: XGB OOF → LASSO manually (stacks requires ≥2 candidates to blend).
    # cv.glmnet selects penalty on the OOF predictions; then XGB is refit on full
    # training data, and the LASSO meta-learner is refit at that penalty on full-train
    # XGB outputs — same two-stage structure stacks uses internally.
    xgb_oof   <- collect_predictions(xgb_res_exp) |> dplyr::arrange(.row)
    x_oof_b   <- matrix(xgb_oof$.pred_1, ncol = 1)
    y_oof_b   <- as.integer(xgb_oof$outcome) - 1L
    cv_b      <- glmnet::cv.glmnet(x_oof_b, y_oof_b, family = "binomial", alpha = 1)

    xgb_fit_b  <- fit(xgb_wf_exp, data = train)
    x_train_b  <- matrix(predict(xgb_fit_b, new_data = train, type = "prob")$.pred_1, ncol = 1)
    x_test_b   <- matrix(predict(xgb_fit_b, new_data = test,  type = "prob")$.pred_1, ncol = 1)
    lasso_b    <- glmnet::glmnet(x_train_b, as.integer(train$outcome) - 1L,
                                 family = "binomial", alpha = 1, lambda = cv_b$lambda.min)
    prob_b     <- as.vector(predict(lasso_b, x_test_b, type = "response"))
    auc_b      <- roc_auc(tibble(.pred_1 = prob_b, outcome = test$outcome),
                          truth = outcome, .pred_1, event_level = "second")$.estimate

    result <- tibble(seed = seed, auc_lr_xgb = auc_a, auc_xgb_only = auc_b)
    saveRDS(result, seed_cache)
    result
  }

  sens_results <- map_dfr(seeds, run_one_seed)
  saveRDS(sens_results, stack_experiment_cache)
}

# Paired t-test: pairing on seed controls for split-level variance,
# isolating the effect of adding LR as a level-1 candidate.
print(t.test(sens_results$auc_lr_xgb, sens_results$auc_xgb_only, paired = TRUE))
cat(sprintf("Mean AUC difference (LR+XGB - XGB only): %.4f\n",
            mean(sens_results$auc_lr_xgb) - mean(sens_results$auc_xgb_only)))

sens_results |>
  tidyr::pivot_longer(
    cols      = c(auc_lr_xgb, auc_xgb_only),
    names_to  = "condition",
    values_to = "auc"
  ) |>
  mutate(condition = dplyr::recode(condition,
    auc_lr_xgb   = "LR + XGB → LASSO",
    auc_xgb_only = "XGB only → LASSO"
  )) |>
  ggplot(aes(x = auc, fill = condition)) +
  geom_density(alpha = 0.5) +
  geom_rug(aes(color = condition), alpha = 0.7) +
  labs(
    title    = "Seed sensitivity: LR+XGB→LASSO vs XGB→LASSO",
    subtitle = sprintf("Paired t-test across %d seeds", nrow(sens_results)),
    x = "Test ROC AUC", y = "Density",
    fill = NULL, color = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top")


# Look at Variable Importance ------
# Permutation importance IS applicable — predict(stack_fit, ...) accepts raw
# features and runs the full two-level ensemble internally. Commented out because
# each permutation runs both XGBoost and logistic inference end-to-end for every
# predictor, making runtime comparable to the standalone XGBoost permutation (~hours).
# Native gain-based importance is NOT available: the meta-learner only sees OOF
# probability columns as its features, not the original predictors.
# Uncomment and run overnight if needed; result cached to stack_vip.rds.

# stack_auc_metric <- function(truth, estimate) {
#   roc_auc_vec(truth = truth, estimate = estimate, event_level = "second")
# }
#
# stack_vip_cache <- here::here("vignettes/stack_vip.rds")
# if (file.exists(stack_vip_cache)) {
#   stack_vip_obj <- readRDS(stack_vip_cache)
# } else {
#   set.seed(12345)
#   stack_vip_obj <- vi_permute(
#     object        = stack_fit,
#     feature_names = setdiff(names(train_data_wopv), c("who", "outcome")),
#     train         = train_data_wopv,
#     target        = "outcome",
#     metric        = stack_auc_metric,
#     smaller_is_better = FALSE,
#     pred_wrapper  = function(object, newdata) {
#       predict(object, new_data = newdata, type = "prob")$.pred_1
#     },
#     nsim = 1
#   )
#   saveRDS(stack_vip_obj, stack_vip_cache)
# }
#
# vip(stack_vip_obj, num_features = 15, geom = "col") +
#   labs(
#     title    = "Stack permutation variable importance",
#     subtitle = "Mean drop in test ROC AUC when each predictor is shuffled (1 permutation)",
#     x        = "Importance (mean ROC AUC decrease)",
#     y        = NULL
#   ) +
#   theme_minimal(base_size = 11)
