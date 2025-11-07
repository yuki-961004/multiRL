# multiRL

<!-- badges: start -->
[![R-CMD-check](https://github.com/yuki-961004/multiRL/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/yuki-961004/multiRL/actions/workflows/R-CMD-check.yaml)
[![Code Coverage](https://codecov.io/gh/yuki-961004/multiRL/graph/badge.svg)](https://app.codecov.io/gh/yuki-961004/multiRL)
<!-- badges: end -->

## Overview

This package is designed to help users build the **Rescorla-Wagner Model** for **Multi-Armed Bandit** tasks (for TAFC see [binaryRL](https://yuki-961004.github.io/binaryRL/)). Beginners can define models using simple **`if-else`** logic, making model construction more accessible.  

* [Step 1](./articles/multiRL.html#id_1-run-model): Build Reinforcement Learning Models `run_m()`
* ~~[Step 2](./articles/multiRL.html#id_2-recovery): Parameter and Model Recovery `rcv_d()`~~
* ~~[Step 3](./articles/multiRL.html#id_3-fit-real-data): Fit Real Data `fit_p()`~~
* ~~[Step 4](./articles/multiRL.html#id_4-replay-the-experiment): Replay the Experiment `rpl_e()`~~ 

<!---------------------------------------------------------->

## Highlights

> 1. Adherence to R S4 Methods.  
> 2. Compatibility with Two-Alternative Forced Choice (TAFC) tasks.  
> 3. Supports Definition of Latent Rules.  
> 4. Array-based Results for Seamless Integration with TensorFlow and ABC.  


## Installation

```r
# Install the latest version from GitHub
remotes::install_github("yuki-961004/multiRL@*release")
# Load package
library(multiRL)
```

## Demo

## Step 1: run_m
```r
multiRL.model <- multiRL::run_m(
  data = multiRL::TAB[multiRL::TAB[, "Subject"] == 1, ],
  behrule = list(
    cue = c("A", "B", "C", "D"),
    rsp = c("A", "B", "C", "D")
  ),
  colnames = list(
    subid = "Subject", block = "Block", trial = "Trial",
    object = c("L_choice", "R_choice"), 
    reward = c("L_reward", "R_reward"),
    action = "Sub_Choose"
  ),
  params = list(
    free = list(
      alphaN = 0.123,
      alphaP = 0.456,
      beta = 0.789
    ),
    fixed = list(
      gamma = 1,
      delta = 0.1,
      epsilon = NA_real_,
      zeta = 1,
      eta = NA_real_
    ),
    constant = list(
      Q1 = NA_real_,
      lapse = 0.01
    )
  ),
  priors = list(
    alphaN = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
    alphaP = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
    beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
  ),
  settings = list(
    name = "RSTD",
    mode = "fitting",
    estimate = "MLE",
    policy = "off"
  ),
  anythingelse = c(1, 2, 3)
)

multiRL.summary <- multiRL::summary(multiRL.model)
```

```
Model Fit:
  Accuracy: 100%
  Log-Likelihood: -671.03
  Log-Prior Probability: -0.83
  Log-Posterior Probability: -671.86
  AIC: 1346.06
  BIC: 1353.84
```

## Estimation
```r
data <- multiRL::TAB |> dplyr::filter(Subject %in% 1:10)
behrule <- list(cue = c("A", "B", "C", "D"), rsp = c("A", "B", "C", "D"))
colnames <- list(
  object = c("L_choice", "R_choice"), 
  reward = c("L_reward", "R_reward"),
  action = "Sub_Choose"
)
models <- list(multiRL::TD, multiRL::RSTD, multiRL::Utility)
settings <- list(list(name = "TD"), list(name = "RSTD"), list(name = "Utility"))
```

### MLE
```r
result.MLE <- multiRL::estimate_1_MLE(
  data = data,
  behrule = settings,
  colnames = colnames,
  models = models,
  settings = settings,
  algorithm = c("NLOPT_GN_MLSL", "NLOPT_LN_BOBYQA"),
  lowers = list(c(0, 0), c(0, 0, 0), c(0, 0, 0)),
  uppers = list(c(1, 1), c(1, 1, 1), c(1, 1, 1)),
  control = list(core = 10, iter = 10)
)
```

### MAP
```r
result.MAP <- multiRL::estimate_1_MAP(
  data = data,
  behrule = behrule,
  colnames = colnames,
  models = models,
  settings = settings,
  priors = list(
    TD = list(
      alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
    ),
    RSTD = list(
      alphaN = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      alphaP = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
    ),
    Utility = list(
      alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)},
      gamma = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}
    )
  ),
  algorithm = c("NLOPT_GN_MLSL", "NLOPT_LN_BOBYQA"),
  lowers = list(c(0, 0), c(0, 0, 0), c(0, 0, 0)),
  uppers = list(c(1, 1), c(1, 1, 1), c(1, 1, 1)),
  control = list(core = 10, iter = c(10, 10))
)
```

### RNN
```r
result.RNN <- multiRL::estimate_2_RNN(
  data = data,
  behrule = behrule,
  colnames = colnames,
  models = models,
  settings = settings,
  priors = list(
    list(
      alpha = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}, 
      beta = function(x) {stats::rexp(n = 1, rate = 1)}
    ),
    list(
      alphaN = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}, 
      alphaP = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}, 
      beta = function(x) {stats::rexp(n = 1, rate = 1)}
    ),
    list(
      alpha = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}, 
      beta = function(x) {stats::rexp(n = 1, rate = 1)},
      gamma = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}
    )
  ),
  control = list(sample = 100, epochs = 10)
)
```
### ABC
```r
```

## Step 2: rcv_d

## Step 3: fit_p

## Step 4: rpl_e