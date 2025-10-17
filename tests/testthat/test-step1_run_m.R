# r
testthat::test_that("run_m", {
  
  filepath <- testthat::test_path("testdata", "data.csv")
  data <- utils::read.csv(filepath)
  
  multiRL.model <- run_m(
    data = data[data[, "Subject"] == 1, ],
    colnames = list(
      subid = "Subject", 
      block = "Block",
      trial = "Trial",
      
      object = c("Object_1", "Object_2", "Object_3", "Object_4"), 
      reward = c("Reward_1", "Reward_2", "Reward_3", "Reward_4"),
      action = "Action"
    ),
    params = list(
      fixed = list(
        Q1 = 0,
        gamma = 1,
        delta = 0.1,
        epsilon = 0.1,
        zeta = 1,
        eta = NA_real_
      ),
      free = list(
        alpha = c(0.3, 0.7),
        beta = 0.5
      )
    ),
    funcs = list(
      rate_func = func_alpha,
      prob_func = func_beta,
      util_func = func_gamma,
      bias_func = func_delta,
      expl_func = func_epsilon
    ),
    behrule = list(
      cue = c("Red", "Yellow", "Green", "Blue"),
      rsp = c("Up", "Down", "Left", "Right")
    ),
    whatyousay = c(1, 2, 3)
  )
  
  multiRL.summary <- summary(multiRL.model)
  
  testthat::expect_s4_class(multiRL.model, "multiRL.model")
  testthat::expect_s4_class(multiRL.summary, "multiRL.summary")
})
