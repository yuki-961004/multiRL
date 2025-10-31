testthat::test_that("MLE", {
  result.MLE <- estimate_1_MLE(
    data = multiRL::TAB %>% dplyr::filter(Subject %in% 1:4),
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
        alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
        beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
      ),
      list(
        alphaN = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
        alphaP = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
        beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
      ),
      list(
        alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
        beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)},
        gamma = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}
      )
    ),
    settings = list(
      list(name = "TD"),
      list(name = "RSTD"),
      list(name = "Utility")
    ),
    
    algorithm = c("NLOPT_GN_MLSL", "NLOPT_LN_BOBYQA"),
    lowers = list(c(0, 0), c(0, 0, 0), c(0, 0, 0)),
    uppers = list(c(1, 1), c(1, 1, 1), c(1, 1, 1)),
    control = list(core = 4)
  )
  
  testthat::expect_s3_class(result.MLE, "data.frame")
})

