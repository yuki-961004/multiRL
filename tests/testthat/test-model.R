# TD
testthat::test_that("TD", {
  
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
    algorithm = c("NLOPT_GN_MLSL", "NLOPT_LN_BOBYQA"),
    lower = c(0, 0),
    upper = c(1, 1),
    iteration = 10,
  )
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
})

# RSTD
testthat::test_that("RSTD", {
  
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
      alpha_n = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      alpha_p = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
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
    model = multiRL::RSTD,
    env = multiRL.env,
    algorithm = c("NLOPT_GN_MLSL", "NLOPT_LN_BOBYQA"),
    lower = c(0, 0, 0),
    upper = c(1, 1, 1),
    iteration = 10,
  )
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
})

# Utility
testthat::test_that("Utility", {
  
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
      name = "Utility",
      mode = "fitting",
      estimate = "MLE",
      policy = "on"
    ),
  )
  
  multiRL.model <- multiRL::estimate_1_LBI(
    model = multiRL::Utility,
    env = multiRL.env,
    algorithm = c("NLOPT_GN_MLSL", "NLOPT_LN_BOBYQA"),
    lower = c(0, 0, 0),
    upper = c(1, 1, 1),
    iteration = 10,
  )
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
})
