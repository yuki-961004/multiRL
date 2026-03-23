# Generating
testthat::test_that("ABC", {
  
  result.ABC <- multiRL::estimation_methods(
    estimate = "ABC",
    data = multiRL::TAB,
    ids = 1:4,
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
    settings = list(name = c("TD", "RSTD", "Utility")),
    
    lowers = list(c(0, 0), c(0, 0, 0), c(0, 0, 0)),
    uppers = list(c(1, 5), c(1, 1, 5), c(1, 1, 5)),
    control = list(
      core = 4, sample = 100, train = 100,
      tol = 0.1, reduction = "NONE"
    )
  )

  testthat::expect_s3_class(result.ABC, "data.frame")
})

# Generating
testthat::test_that("ABC", {
  
  result.ABC <- multiRL::estimation_methods(
    estimate = "ABC",
    data = multiRL::TAB,
    ids = 1:4,
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
    settings = list(name = c("TD", "RSTD", "Utility")),
    
    lowers = list(c(0, 0), c(0, 0, 0), c(0, 0, 0)),
    uppers = list(c(1, 5), c(1, 1, 5), c(1, 1, 5)),
    control = list(
      core = 4, sample = 100, train = 100,
      tol = 0.1, reduction = "PLS"
    )
  )
  
  testthat::expect_s3_class(result.ABC, "data.frame")
})

# Generating
testthat::test_that("ABC", {
  
  result.ABC <- multiRL::estimation_methods(
    estimate = "ABC",
    data = multiRL::TAB,
    ids = 1:4,
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
    settings = list(name = c("TD", "RSTD", "Utility")),
    
    lowers = list(c(0, 0), c(0, 0, 0), c(0, 0, 0)),
    uppers = list(c(1, 5), c(1, 1, 5), c(1, 1, 5)),
    control = list(
      core = 4, sample = 100, train = 100,
      tol = 0.1, reduction = "PCA"
    )
  )
  
  testthat::expect_s3_class(result.ABC, "data.frame")
})