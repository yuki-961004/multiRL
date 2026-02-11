testthat::test_that("MAP", {
  
  result.MAP <- multiRL::estimation_methods(
    estimate = "MAP",
    # 数据
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
    # 模型
    models = list(multiRL::TD, multiRL::RSTD, multiRL::Utility),
    priors = list(
      list(
        alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
        beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
      ),
      list(
        alphaN = function(x) {stats::dunif(x, min = 0, max = 1, log = TRUE)}, 
        alphaP = function(x) {stats::dnorm(x, mean = 0.5, sd = 0.1, log = TRUE)},
        beta = function(x) {stats::dlnorm(x, meanlog = 0.5, sdlog = 0.1, log = TRUE)}
      ),
      list(
        alpha = function(x) {stats::dgamma(x, shape = 2, rate = 3, log = TRUE)}, 
        beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)},
        gamma = function(x) {stats::dlogis(x, location = 0, scale = 1, log = TRUE)}
      )
    ),
    settings = list(name = c("TD", "RSTD", "Utility")),
    # 算法
    algorithm = "NLOPT_GN_MLSL",
    lowers = list(c(0, 0), c(0, 0, 0), c(0, 0, 0)),
    uppers = list(c(1, 1), c(1, 1, 1), c(1, 1, 1)),
    control = list(core = 4, iter = 10, patience = 1)
  )
  
  testthat::expect_s3_class(result.MAP, "data.frame")
})

