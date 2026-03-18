# MLE
testthat::test_that("MLE", {
  
  recovery.MLE <- multiRL::rcv_d(
    estimate = "MLE",
    data = multiRL::TAB,
    id = 1,
    behrule = list(
      cue = c("A", "B", "C", "D"),
      rsp = c("A", "B", "C", "D")
    ),
    
    colnames = list(
      object = c("L_choice", "R_choice"), 
      reward = c("L_reward", "R_reward"),
      action = "Sub_Choose"
    ),
    models = list(multiRL::TD, multiRL::RSTD, multiRL::Utility),
    settings = list(list(name = "TD"), list(name = "RSTD"), list(name = "Utility")),
    priors = list(
      list(
        alpha = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}, 
        beta = function(x) {stats::rexp(n = 1, rate = 1)}
      ),
      list(
        alphaN = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}, 
        alphaP = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}, 
        beta = function(x) {stats::rexp(n = 1, rate = 1)}
      ),
      list(
        alpha = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}, 
        beta = function(x) {stats::rexp(n = 1, rate = 1)},
        gamma = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}
      )
    ),
    
    algorithm = c("NLOPT_GN_MLSL", "NLOPT_LN_BOBYQA"),
    lowers = list(c(0, 0), c(0, 0, 0), c(0, 0, 0)),
    uppers = list(c(1, 1), c(1, 1, 1), c(1, 1, 1)),
    control = list(core = 1, sample = 5, iter = 5)
  )

  
  testthat::expect_s3_class(recovery.MLE, "multiRL.recovery")
  
})

# ABC
testthat::test_that("ABC", {
  
  recovery.ABC <- multiRL::rcv_d(
    estimate = "ABC",
    data = multiRL::TAB,
    id = 1,
    behrule = list(
      cue = c("A", "B", "C", "D"),
      rsp = c("A", "B", "C", "D")
    ),
    
    colnames = list(
      object = c("L_choice", "R_choice"), 
      reward = c("L_reward", "R_reward"),
      action = "Sub_Choose"
    ),
    models = list(multiRL::TD, multiRL::RSTD, multiRL::Utility),
    settings = list(list(name = "TD"), list(name = "RSTD"), list(name = "Utility")),
    priors = list(
      list(
        alpha = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}, 
        beta = function(x) {stats::rexp(n = 1, rate = 1)}
      ),
      list(
        alphaN = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}, 
        alphaP = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}, 
        beta = function(x) {stats::rexp(n = 1, rate = 1)}
      ),
      list(
        alpha = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}, 
        beta = function(x) {stats::rexp(n = 1, rate = 1)},
        gamma = function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}
      )
    ),
    
    lowers = list(c(0, 0), c(0, 0, 0), c(0, 0, 0)),
    uppers = list(c(1, 5), c(1, 1, 5), c(1, 1, 5)),
    control = list(tol = 0.5, core = 1, sample = 5, train = 100)
  )
  
  testthat::expect_s3_class(recovery.ABC, "multiRL.recovery")
  
})
