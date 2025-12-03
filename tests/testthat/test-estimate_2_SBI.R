# Generating
testthat::test_that("SBI", {
  
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
    settings = list(
      name = "TD",
      mode = "simulating",
      estimate = "ABC",
      policy = "on"
    ),
  )
  
  list_simulated <- multiRL::estimate_2_SBI(
    model = multiRL::TD,
    env = multiRL.env,
    priors = list(
      alpha = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}, 
      beta = function(x) {stats::rexp(n = 1, rate = 1)}
    ),
    control = list(sample = 10)
  )

  testthat::expect_type(list_simulated, "list")
})

