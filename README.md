# ctn0094.oliven

**Predicting treatment response for people undergoing care for opioid use disorder (OUD)**

**Author**: Nat Oliven (<oliven@umich.edu>, ORCID [0009-0005-2359-319X](https://orcid.org/0009-0005-2359-319X))

**Description**: An R analysis package that builds a pre-treatment feature matrix
for care-seeking adults with OUD and benchmarks a range of models — from an
interpretable logistic-regression baseline through tree ensembles and a
pretrained transformer — for predicting relapse during treatment.

**Data source**: All data are derived from the public, de-identified harmonized
NIDA Clinical Trials Network datasets distributed in the
[`public.ctn0094data`](https://cran.r-project.org/package=public.ctn0094data)
and [`CTNote`](https://cran.r-project.org/package=CTNote) R packages, pooling
three randomized MOUD trials — CTN-0027, CTN-0030, and CTN-0051 (2006–2016).
No raw or confidential participant data are stored in this repository.

**Notes**: Progress notes are in [dated_progress_notes.md](dated_progress_notes.md).

## Run Me First

```r
install.packages("remotes")

# Harmonized CTN data (CRAN)
remotes::install_cran(c("public.ctn0094data", "CTNote"))

# Core analysis stack
remotes::install_cran(
  c("conflicted", "tidyverse", "tidymodels", "glmnet", "kknn", "rpart",
    "rpart.plot", "ranger", "xgboost", "vip", "gtsummary", "knitr",
    "quarto", "digest", "scales", "xfun")
)
```

The TabPFN model additionally requires the Python `tabpfn` package, accessed
from R via `reticulate` in a virtual environment named `tabpfn-env`.

## The Analysis

The full write-up — abstract, methods, results, and discussion — is the
`analysis.qmd` vignette. Render it with:

```r
quarto::quarto_render("vignettes/analysis.qmd")
```

It constructs a feature matrix from the 28 days before randomization
(days −28 to −1) and evaluates whether pre-treatment behavioral, substance-use,
demographic, and clinical features predict relapse. Models are selected on
5-fold cross-validated ROC AUC and confirmed on a held-out 25% test split, all
sharing one leakage-safe `recipes` preprocessing pipeline (treatment-arm proxies
removed, tiered missingness handling, normalization, correlation pruning).

### Models benchmarked

| Model | Engine | Notes |
|-------|--------|-------|
| Logistic regression | `glm` | Interpretable baseline; full-predictor and "no problem variables" (separation-handling) variants |
| LASSO / elastic net | `glmnet` | Penalized logistic; 1-SE rule for sparsity |
| K-nearest neighbors | `kknn` | Non-parametric; tuned over neighbor count |
| CART | `rpart` | Single pruned decision tree |
| Random forest | `ranger` | 1,000-tree ensemble; impurity importance |
| Gradient boosting | `xgboost` | Latin-hypercube tuning with early stopping |
| TabPFN | `tabpfn` (Python) | Pretrained transformer; no tuning |

## Dataset

`analysis_tibble` — a tibble of **2,492 participants × 116 columns** shipped with
the package (one row per participant: an ID, source study, the modeled relapse
outcome, and engineered pre-treatment predictors). See `?analysis_tibble` for
the column dictionary.

```r
library(ctn0094.oliven)
data(analysis_tibble)
```

## Ordered Analysis Pipeline

1. `data-raw/analysis_tibble.R` — builds the feature matrix (sources
   `vignettes/building_analysis.R`) and saves `data/analysis_tibble.rda`.
2. `vignettes/building_analysis.R` — feature engineering (drug-use aggregation,
   demographics, severity scores, interactions).
3. Per-model scripts sourced by the vignette, in order:
   `logistic_regression.R`, `logistic_regression_without_problem_vars.R`,
   `logistic_regression_wopv_kfold.R`, `logistic_enet.R`, `knn.R`,
   `knn_wopv.R`, `cart.R`, `random_forest.R`, `xgboost.R`, `tabpfn.R`.
4. `vignettes/analysis.qmd` — renders the full report, sourcing the scripts above.

Supporting files: `metrics_helpers.R` (shared metric utilities),
`drug_mapping_diagram.R` (the feature-aggregation figure),
`stack_consist_test_log_xgb.R` (seed-sensitivity stacking experiment).

## Directories

- `R/` — package code and dataset documentation.
- `data/`, `data-raw/` — the packaged dataset and the script that builds it.
- `vignettes/` — the analysis report, per-model scripts, and figures.
- `inst/slides/` — a reveal.js results deck (`final_results.qmd`).
- `inst/docs/`, `inst/scratch/` — feature-selection notes and exploratory code.

## License

See [LICENSE](LICENSE). Currently "all rights reserved" — add an explicit
open-source license before publishing if you intend others to reuse the code.
