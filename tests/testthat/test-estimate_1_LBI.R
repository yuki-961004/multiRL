# L-BFGS-B
testthat::test_that("L-BFGS-B", {
  
  data <- multiRL::TAB
  
  multiRL.env <- multiRL::estimate_0_ENV(
    data = data[data[, "Subject"] == 1, ],
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
    settings = list(name = "TD"),
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
  
  data <- multiRL::TAB
  
  multiRL.env <- multiRL::estimate_0_ENV(
    data = data[data[, "Subject"] == 1, ],
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
    settings = list(name = "TD"),
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
  
  data <- multiRL::TAB
  
  multiRL.env <- multiRL::estimate_0_ENV(
    data = data[data[, "Subject"] == 1, ],
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
    settings = list(name = "TD"),
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
  
  data <- multiRL::TAB
  
  multiRL.env <- multiRL::estimate_0_ENV(
    data = data[data[, "Subject"] == 1, ],
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
    settings = list(name = "TD"),
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
  
  data <- multiRL::TAB
  
  multiRL.env <- multiRL::estimate_0_ENV(
    data = data[data[, "Subject"] == 1, ],
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
    settings = list(name = "TD"),
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
  
  data <- multiRL::TAB
  
  multiRL.env <- multiRL::estimate_0_ENV(
    data = data[data[, "Subject"] == 1, ],
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
    settings = list(name = "TD"),
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
  
  data <- multiRL::TAB
  
  multiRL.env <- multiRL::estimate_0_ENV(
    data = data[data[, "Subject"] == 1, ],
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
    settings = list(name = "TD"),
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

# NLOPT
testthat::test_that("NLOPT", {
  
  data <- multiRL::TAB
  
  multiRL.env <- multiRL::estimate_0_ENV(
    data = data[data[, "Subject"] == 1, ],
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
    settings = list(name = "TD"),
  )
  
  multiRL.model <- multiRL::estimate_1_LBI(
    model = multiRL::TD,
    env = multiRL.env,
    algorithm = c("NLOPT_GN_MLSL", "NLOPT_LN_BOBYQA"),
    lower = c(0, 0),
    upper = c(1, 1),
    iteration = 5,
  )
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
})