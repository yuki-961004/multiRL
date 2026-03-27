# TD
testthat::test_that("TD", {
  
  multiRL.env <- multiRL::estimate_0_ENV(
    data = multiRL::TAB[multiRL::TAB[, "Subject"] == 1, ],
    behrule = list(
      cue = c("A", "B", "C", "D"),
      rsp = c("A", "B", "C", "D")
    ),
    colnames = list(
      object = c("L_choice", "R_choice"), 
      reward = c("L_reward", "R_reward"),
      action = "Sub_Choose"
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
    lower = c(0, 0),
    upper = c(1, 1),
    control = list(iter = 5)
  )
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
})

# RSTD
testthat::test_that("RSTD", {
  
  multiRL.env <- multiRL::estimate_0_ENV(
    data = multiRL::TAB[multiRL::TAB[, "Subject"] == 1, ],
    behrule = list(
      cue = c("A", "B", "C", "D"),
      rsp = c("A", "B", "C", "D")
    ),
    colnames = list(
      object = c("L_choice", "R_choice"), 
      reward = c("L_reward", "R_reward"),
      action = "Sub_Choose"
    ),
    priors = list(
      alphaN = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      alphaP = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
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
    lower = c(0, 0, 0),
    upper = c(1, 1, 1),
    control = list(iter = 5)
  )
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
})

# Utility
testthat::test_that("Utility", {
  
  multiRL.env <- multiRL::estimate_0_ENV(
    data = multiRL::TAB[multiRL::TAB[, "Subject"] == 1, ],
    behrule = list(
      cue = c("A", "B", "C", "D"),
      rsp = c("A", "B", "C", "D")
    ),
    colnames = list(
      object = c("L_choice", "R_choice"), 
      reward = c("L_reward", "R_reward"),
      action = "Sub_Choose"
    ),
    priors = list(
      alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)},
      gamma = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}
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
    lower = c(0, 0, 0),
    upper = c(1, 1, 1),
    control = list(iter = 5)
  )
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
})
