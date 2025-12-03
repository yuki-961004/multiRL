# multiRL <a href="https://yuki-961004.github.io/multiRL/"><img src="./fig/logo.png" alt="LOGO" align="right" width="120"/></a>

<!-- badges: start -->
[![R-CMD-check](https://github.com/yuki-961004/multiRL/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/yuki-961004/multiRL/actions/workflows/R-CMD-check.yaml)
[![Code Coverage](https://codecov.io/gh/yuki-961004/multiRL/graph/badge.svg)](https://app.codecov.io/gh/yuki-961004/multiRL)
<!-- badges: end -->

## Overview

This package is designed to help users build the **Rescorla-Wagner Model** for **Multi-Armed Bandit** tasks (for TAFC see [binaryRL](https://yuki-961004.github.io/binaryRL/)). Beginners can define models using simple **`if-else`** logic, making model construction more accessible.  

* [Step 1](./articles/multiRL.html#id_1-run-model): Build Reinforcement Learning Models `run_m()`
* [Step 2](./articles/multiRL.html#id_2-recovery): Parameter and Model Recovery `rcv_d()`
* [Step 3](./articles/multiRL.html#id_3-fit-real-data): Fit Real Data `fit_p()`
* [Step 4](./articles/multiRL.html#id_4-replay-the-experiment): Replay the Experiment `rpl_e()`

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
    free = list(alphaN = 0.3, alphaP = 0.7, beta = 0.5),
    fixed = list(
      gamma = 1, delta = 0.1, epsilon = NA_real_, 
      zeta = 1, eta = NA_real_, theta = 0
    ),
    constant = list(Q1 = NA_real_, lapse = 0.01, bonus = 0)
  ),
  priors = list(
    alphaN = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
    alphaP = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
    beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
  ),
  settings = list(
    name = "RSTD", mode = "fitting", estimate = "MLE", policy = "off"
  )
)

multiRL.summary <- multiRL::summary(multiRL.model)
```

```
Model Fit:
  Accuracy: 100%
  Log-Likelihood: -384.01
  Log-Prior Probability: -0.04
  Log-Posterior Probability: -384.05
  AIC: 774.03
  BIC: 785.69
```

## Arguments

```r
behrule = list(
  cue = c("A", "B", "C", "D"),
  rsp = c("A", "B", "C", "D")
)

colnames = list(
  object = c("L_choice", "R_choice"), 
  reward = c("L_reward", "R_reward"),
  action = "Sub_Choose"
)
models = list(multiRL::TD, multiRL::RSTD, multiRL::Utility)

settings = list(list(name = "TD"), list(name = "RSTD"), list(name = "Utility"))

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
)

algorithm = c("NLOPT_GN_MLSL", "NLOPT_LN_BOBYQA")
lowers = list(c(0, 0), c(0, 0, 0), c(0, 0, 0))
uppers = list(c(1, 1), c(1, 1, 1), c(1, 1, 1))
control = list(...)
```

## Step 2: rcv_d

```r
recovery <- multiRL::rcv_d(
  estimate = c("MLE", "MAP", "ABC", "RNN"),
  data = multiRL::TAB,
  behrule = behrule,
  colnames = colnames,
  models = models,
  priors = priors,
  settings = settings,
  algorithm = algorithm,
  lowers = lowers,
  uppers = uppers,
  control = control
)
```

## Step 3: fit_p

```r
fitting <- multiRL::fit_p(
  estimate = c("MLE", "MAP", "ABC", "RNN"),
  data = multiRL::TAB,
  behrule = behrule,
  colnames = colnames,
  models = models,
  priors = priors,
  settings = settings,
  algorithm = algorithm,
  lowers = lowers,
  uppers = uppers,
  control = control
)
```

## Step 4: rpl_e

```r
replay <- multiRL::rpl_e(
  result = fitting,
  data = multiRL::TAB,
  behrule = behrule,
  colnames = colnames,
  models = models,
  settings = settings,
  priors = priors
)
```