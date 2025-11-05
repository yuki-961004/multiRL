# Generating
testthat::test_that("SBI", {
  multiRL.env <- estimate_0_ENV(
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
    settings = list(
      name = "TD",
      mode = "simulating",
      estimate = "ABC",
      policy = "on"
    ),
  )
  
  list_simulated <- estimate_2_SBI(
    model = multiRL::TD,
    env = multiRL.env,
    priors = list(
      alpha = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}, 
      beta = function(x) {stats::rexp(n = 1, rate = 1)}
    ),
    control = list(iter = 10)
  )

  testthat::expect_type(list_simulated, "list")
})

