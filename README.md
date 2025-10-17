# multiRL

<!-- badges: start -->

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
library(binaryRL)
```

## Demo

```r
multiRL.model <- multiRL::run_m(
  data = binaryRL::Mason_2024_G2 |> dplyr::filter(Subject == 1),
  colnames = list(
    subid = "Subject", 
    block = "Block",
    trial = "Trial",
    object = c("L_choice", "R_choice"), 
    reward = c("L_reward", "R_reward"),
    action = "Sub_Choose"
  ),
  params = list(
    fixed = list(
      Q1 = 100,
      gamma = 1,
      delta = 0.1,
      epsilon = NA_real_,
      zeta = 1,
      eta = NA_real_
    ),
    free = list(
      alpha = c(0.123, 0.456),
      beta = 0.789
    )
  ),
  funcs = list(
    rate_func = multiRL::func_alpha,
    prob_func = multiRL::func_beta,
    util_func = multiRL::func_gamma,
    bias_func = multiRL::func_delta,
    expl_func = multiRL::func_epsilon
  ),
  behrule = list(
    cue = c("A", "B", "C", "D"),
    rsp = c("A", "B", "C", "D")
  ),
  anythingelse = c(1, 2, 3)
)

result <- summary(multiRL.model)
```

```
Model Fit:
  Accuracy: 54.44%
  Log-Likelihood: -484.81
  Log-Prior Probability: 
  Log-Posterior Probability: 
  AIC: 973.61
  BIC: 981.39
```