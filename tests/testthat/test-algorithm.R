# 
testthat::test_that("L-BFGS-B", {
  
  multiRL.env <- estimate_0_ENV(
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
    funcs = list(
      rate_func = multiRL::func_alpha,
      prob_func = multiRL::func_beta,
      util_func = multiRL::func_gamma,
      bias_func = multiRL::func_delta,
      expl_func = multiRL::func_epsilon
    ),
    priors = list(
      alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
    ),
    settings = list(
      name = "TD",
      mode = "fitting",
      estimate = "MLE",
      policy = "on"
    ),
  )
  
  multiRL.model <- multiRL::estimate_1_LBI(
    model = multiRL::TD,
    env = multiRL.env,
    algorithm = "L-BFGS-B",
    lower = c(0, 0),
    upper = c(1, 1),
    iteration = 5,
  )
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
})

# GenSA
testthat::test_that("GenSA", {
  
  multiRL.env <- estimate_0_ENV(
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
    funcs = list(
      rate_func = multiRL::func_alpha,
      prob_func = multiRL::func_beta,
      util_func = multiRL::func_gamma,
      bias_func = multiRL::func_delta,
      expl_func = multiRL::func_epsilon
    ),
    priors = list(
      alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
    ),
    settings = list(
      name = "TD",
      mode = "fitting",
      estimate = "MLE",
      policy = "on"
    ),
  )
  
  multiRL.model <- multiRL::estimate_1_LBI(
    model = multiRL::TD,
    env = multiRL.env,
    algorithm = "GenSA",
    lower = c(0, 0),
    upper = c(1, 1),
    iteration = 5,
  )
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
})

# GA
testthat::test_that("GA", {
  
  multiRL.env <- estimate_0_ENV(
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
    funcs = list(
      rate_func = multiRL::func_alpha,
      prob_func = multiRL::func_beta,
      util_func = multiRL::func_gamma,
      bias_func = multiRL::func_delta,
      expl_func = multiRL::func_epsilon
    ),
    priors = list(
      alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
    ),
    settings = list(
      name = "TD",
      mode = "fitting",
      estimate = "MLE",
      policy = "on"
    ),
  )
  
  multiRL.model <- multiRL::estimate_1_LBI(
    model = multiRL::TD,
    env = multiRL.env,
    algorithm = "GA",
    lower = c(0, 0),
    upper = c(1, 1),
    iteration = 5,
  )
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
})

# DEoptim
testthat::test_that("DEoptim", {
  
  multiRL.env <- estimate_0_ENV(
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
    funcs = list(
      rate_func = multiRL::func_alpha,
      prob_func = multiRL::func_beta,
      util_func = multiRL::func_gamma,
      bias_func = multiRL::func_delta,
      expl_func = multiRL::func_epsilon
    ),
    priors = list(
      alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
    ),
    settings = list(
      name = "TD",
      mode = "fitting",
      estimate = "MLE",
      policy = "on"
    ),
  )
  
  multiRL.model <- multiRL::estimate_1_LBI(
    model = multiRL::TD,
    env = multiRL.env,
    algorithm = "DEoptim",
    lower = c(0, 0),
    upper = c(1, 1),
    iteration = 5,
  )
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
})

# PSO
testthat::test_that("PSO", {
  
  multiRL.env <- estimate_0_ENV(
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
    funcs = list(
      rate_func = multiRL::func_alpha,
      prob_func = multiRL::func_beta,
      util_func = multiRL::func_gamma,
      bias_func = multiRL::func_delta,
      expl_func = multiRL::func_epsilon
    ),
    priors = list(
      alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
    ),
    settings = list(
      name = "TD",
      mode = "fitting",
      estimate = "MLE",
      policy = "on"
    ),
  )
  
  multiRL.model <- multiRL::estimate_1_LBI(
    model = multiRL::TD,
    env = multiRL.env,
    algorithm = "PSO",
    lower = c(0, 0),
    upper = c(1, 1),
    iteration = 5,
  )
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
})

# Bayesian
testthat::test_that("Bayesian", {
  
  multiRL.env <- estimate_0_ENV(
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
    funcs = list(
      rate_func = multiRL::func_alpha,
      prob_func = multiRL::func_beta,
      util_func = multiRL::func_gamma,
      bias_func = multiRL::func_delta,
      expl_func = multiRL::func_epsilon
    ),
    priors = list(
      alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
    ),
    settings = list(
      name = "TD",
      mode = "fitting",
      estimate = "MLE",
      policy = "on"
    ),
  )
  
  multiRL.model <- multiRL::estimate_1_LBI(
    model = multiRL::TD,
    env = multiRL.env,
    algorithm = "Bayesian",
    lower = c(0, 0),
    upper = c(1, 1),
    iteration = 5,
  )
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
})

# CMA-ES
testthat::test_that("CMA-ES", {
  
  multiRL.env <- estimate_0_ENV(
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
    funcs = list(
      rate_func = multiRL::func_alpha,
      prob_func = multiRL::func_beta,
      util_func = multiRL::func_gamma,
      bias_func = multiRL::func_delta,
      expl_func = multiRL::func_epsilon
    ),
    priors = list(
      alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
    ),
    settings = list(
      name = "TD",
      mode = "fitting",
      estimate = "MLE",
      policy = "on"
    ),
  )
  
  multiRL.model <- multiRL::estimate_1_LBI(
    model = multiRL::TD,
    env = multiRL.env,
    algorithm = "CMA-ES",
    lower = c(0, 0),
    upper = c(1, 1),
    iteration = 5,
  )
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
})