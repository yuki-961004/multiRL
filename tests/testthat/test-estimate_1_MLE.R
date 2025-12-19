testthat::test_that("MLE", {

  result.MLE <- multiRL::estimation_methods(
    estimate = "MLE",
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
    settings = list(name = c("TD", "RSTD", "Utility"), policy = "off"),
    # 算法
    algorithm = algorithm = "NLOPT_GN_MLSL",
    lowers = list(c(0, 0), c(0, 0, 0), c(0, 0, 0)),
    uppers = list(c(1, 1), c(1, 1, 1), c(1, 1, 1)),
    control = list(core = 4, iter = 5)
  )
  
  testthat::expect_s3_class(result.MLE, "data.frame")
})

