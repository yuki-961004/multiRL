# r
testthat::test_that("Epsilon-First, non-priors, on-policy", {
  
  data <- multiRL::MAB
  
  multiRL.model <- multiRL::run_m(
    data = data[data[, "Subject"] == 1, ],
    behrule = list(
      cue = c("Red", "Yellow", "Green", "Blue"),
      rsp = c("Up", "Down", "Left", "Right")
    ),
    params = list(
      free = list(
        alpha = c(0.3, 0.7),
        beta = 0.5
      ),
      fixed = list(
        gamma = 1,
        delta = 0.1,
        epsilon = NA_real_,
        zeta = 20,
        eta = NA_real_
      ),
      constant = list(
        Q1 = NA_real_,
        lapse = 0.01
      )
    ),
    priors = list(),
    settings = list(
      name = "TD",
      mode = "fitting",
      estimate = "ABC",
      policy = "on"
    ),
    anythingelse = c(1, 2, 3)
  )
  
  multiRL.summary <- summary(multiRL.model)
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
  testthat::expect_s4_class(multiRL.summary, "multiRL.summary")
})

testthat::test_that("Epsilon-Greedy, priors, on-policy", {
  
  data <- multiRL::MAB
  
  multiRL.model <- multiRL::run_m(
    data = data[data[, "Subject"] == 1, ],
    behrule = list(
      cue = c("Red", "Yellow", "Green", "Blue"),
      rsp = c("Up", "Down", "Left", "Right")
    ),
    params = list(
      free = list(
        alpha = c(0.3, 0.7),
        beta = 0.5
      ),
      fixed = list(
        gamma = 1,
        delta = 0.1,
        epsilon = 0.3,
        zeta = 1,
        eta = NA_real_
      ),
      constant = list(
        Q1 = 0,
        lapse = 0.01
      )
    ),
    priors = list(
      alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
    ),
    settings = list(
      name = "TD",
      mode = "fit",
      estimate = "RNN",
      policy = "on"
    ),
    anythingelse = c(1, 2, 3)
  )
  
  multiRL.summary <- summary(multiRL.model)
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
  testthat::expect_s4_class(multiRL.summary, "multiRL.summary")
})

testthat::test_that("Epsilon-Decreasing, priors, off-policy", {
  
  data <- multiRL::TAB
  
  multiRL.model <- multiRL::run_m(
    data = data[data[, "Subject"] == 1, ],
    behrule = list(
      cue = c("A", "B", "C", "D"),
      rsp = c("A", "B", "C", "D")
    ),
    colnames = list(
      subid = "Subject", 
      block = "Block",
      trial = "Trial",
      object = c("L_choice", "R_choice"), 
      reward = c("L_reward", "R_reward"),
      action = "Sub_Choose"
    ),
    params = list(
      free = list(
        alpha = c(0.3, 0.7),
        beta = 0.5
      ),
      fixed = list(
        gamma = 1,
        delta = 0.1,
        epsilon = NA_real_,
        zeta = 1,
        eta = 0.1
      ),
      constant = list(
        Q1 = NA_real_,
        lapse = 0.01
      )
    ),
    priors = list(
      alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
    ),
    settings = list(
      name = "TD",
      mode = "fit",
      estimate = "MLE",
      policy = "off"
    ),
    anythingelse = c(1, 2, 3)
  )
  
  multiRL.summary <- summary(multiRL.model)
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
  testthat::expect_s4_class(multiRL.summary, "multiRL.summary")
})
