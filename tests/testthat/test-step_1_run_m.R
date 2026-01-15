# R
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
        alphaN = 0.3,
        alphaP = 0.7,
        beta = 0.5
      ),
      fixed = list(
        gamma = 1,
        delta = 0.1,
        epsilon = NA_real_,
        zeta = 0
      ),
      constant = list(
        Q0 = NA_real_,
        reset = 0,
        lapse = 0.01,
        threshold = 20,
        bonus = 0,
        weight = c(0.5, 0.5)
      )
    ),
    priors = list(),
    settings = list(
      name = "RSTD",
      mode = "fitting",
      estimate = "ABC",
      policy = "on",
      system = c("RL", "WM")
    ),
    engine = "R",
    anythingelse = c(1, 2, 3)
  )
  
  multiRL.summary <- summary(multiRL.model)
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
  testthat::expect_s4_class(multiRL.summary, "multiRL.summary")
})

# Cpp
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
        alphaN = 0.3,
        alphaP = 0.7,
        beta = 0.5
      ),
      fixed = list(
        gamma = 1,
        delta = 0.1,
        epsilon = 0.3,
        zeta = 0
      ),
      constant = list(
        Q0 = 0,
        reset = NA_real_,
        lapse = 0.01,
        threshold = 1,
        bonus = 0,
        weight = 1
      )
    ),
    priors = list(
      alphaN = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      alphaP = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
    ),
    settings = list(
      name = "RSTD",
      mode = "fit",
      estimate = "RNN",
      policy = "on",
      system = "RL"
    ),
    anythingelse = c(1, 2, 3),
    engine = "Cpp"
  )
  
  multiRL.summary <- summary(multiRL.model)
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
  testthat::expect_s4_class(multiRL.summary, "multiRL.summary")
})

# Cpp
testthat::test_that("Epsilon-Decreasing, priors, off-policy", {
  
  data <- multiRL::TAB
  
  multiRL.model <- multiRL::run_m(
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
    params = list(
      free = list(
        alphaN = 0.3,
        alphaP = 0.7,
        beta = 0.5
      ),
      fixed = list(
        gamma = 1,
        delta = 0.1,
        epsilon = 0.3,
        zeta = 0
      ),
      constant = list(
        Q0 = 0,
        reset = 0,
        lapse = 0.01,
        threshold = 0,
        bonus = 0,
        weight = 1
      )
    ),
    priors = list(
      alphaN = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      alphaP = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
      beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
    ),
    settings = list(
      name = "RSTD",
      mode = "fit",
      estimate = "MLE",
      policy = "off",
      system = "WM"
    ),
    engine = "Cpp",
    anythingelse = c(1, 2, 3)
  )
  
  multiRL.summary <- summary(multiRL.model)
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
  testthat::expect_s4_class(multiRL.summary, "multiRL.summary")
})
