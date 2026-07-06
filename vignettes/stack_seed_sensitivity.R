library(stacks)
library(tidymodels)
library(xgboost)
library(dplyr)
library(purrr)
library(conflicted)
conflicts_prefer(dplyr::filter)
conflicts_prefer(purrr::discard)

if (!exists("analysis_tibble_wopv")) {
  source(here::here("vignettes/logistic_regression_without_problem_vars.R"))
}

xgboost_tune_loaded <- readRDS(here::here("vignettes/xgboost_tune.rds"))
xgboost_favorite    <- select_best(xgboost_tune_loaded, metric = "roc_auc")

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

seeds <- c(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L,
           111L, 222L, 333L, 444L, 555L, 666L, 777L, 888L, 999L, 12345L)

n_seeds     <- length(seeds)
n_folds     <- 5L
fits_per_seed <- n_folds * 2L   # LR folds + XGB folds (per condition A; reused for B)
total_fits  <- n_seeds * fits_per_seed
fit_count   <- 0L

run_one_seed <- function(seed) {
  cache_path <- here::here(sprintf("vignettes/stack_sens_seed_%d.rds", seed))
  if (file.exists(cache_path)) {
    fit_count <<- fit_count + fits_per_seed
    cat(sprintf("[%d/%d fits] seed %d loaded from cache\n", fit_count, total_fits, seed))
    return(readRDS(cache_path))
  }

  set.seed(seed)
  split <- initial_split(analysis_tibble_wopv, prop = 0.75, strata = outcome)
  train <- training(split)
  test  <- testing(split)
  folds <- vfold_cv(train, v = 5, strata = outcome)

  xgb_wf <- workflow() |>
    add_recipe(lr_recipe_wopv) |>
    add_model(stack_xgb_spec)

  lr_res <- fit_resamples(lr_workflow_wopv, resamples = folds,
                          control = control_stack_resamples())
  fit_count <<- fit_count + n_folds
  cat(sprintf("[%d/%d fits] seed %d — LR resamples done\n", fit_count, total_fits, seed))

  xgb_res <- fit_resamples(xgb_wf, resamples = folds,
                           control = control_stack_resamples())
  fit_count <<- fit_count + n_folds
  cat(sprintf("[%d/%d fits] seed %d — XGB resamples done\n", fit_count, total_fits, seed))

  get_test_auc <- function(stack_data_obj) {
    preds <- predict(fit_members(blend_predictions(stack_data_obj)),
                     new_data = test, type = "prob") |>
      bind_cols(test |> dplyr::select(outcome))
    roc_auc(preds, truth = outcome, .pred_1, event_level = "second")$.estimate
  }

  auc_a <- get_test_auc(stacks() |> add_candidates(lr_res) |> add_candidates(xgb_res))
  auc_b <- get_test_auc(stacks() |> add_candidates(xgb_res))

  result <- tibble(seed = seed, auc_lr_xgb = auc_a, auc_xgb_only = auc_b)
  saveRDS(result, cache_path)
  result
}

sens_results <- map_dfr(seeds, run_one_seed)
sens_results

cat("\n--- Welch's t-test: LR+XGB vs XGB-only ---\n")
print(t.test(sens_results$auc_lr_xgb, sens_results$auc_xgb_only))

cat(sprintf("Mean AUC difference (LR+XGB - XGB only): %.4f\n",
            mean(sens_results$auc_lr_xgb) - mean(sens_results$auc_xgb_only)))
