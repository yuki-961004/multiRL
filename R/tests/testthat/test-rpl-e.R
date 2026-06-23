testthat::local_edition(3)

testthat::skip_if_not_installed("multiRL")

# Pre-extract data for speed
data_sub <- multiRL::TAB[multiRL::TAB[, "Subject"] == 1, ]

testthat::test_that("rpl_e on fit_p result", {
  fit <- fit_p(
    estimator = "mle",
    data = data_sub,
    colnames = list(
      object = c("L_choice", "R_choice"), 
      reward = c("L_reward", "R_reward"),
      action = "Sub_Choose"
    ),
    behrule = list(
      cue = c("A", "B", "C", "D"),
      rsp = c("A", "B", "C", "D")
    ),
    models = list(TD),
    control = list(iter = 5)
  )

  rep_fit <- rpl_e(fit, option = list(plot = TRUE))
  testthat::expect_s3_class(rep_fit, "multiRLcpp_rpl_e")
  testthat::expect_s3_class(rep_fit$plot, "ggplot")

  # Test S3 plot method
  testthat::expect_no_error(plot(rep_fit))
})

testthat::test_that("rpl_e on rcv_d result", {
  rec <- rcv_d(
    estimator = "mle",
    data = data_sub,
    colnames = list(
      object = c("L_choice", "R_choice"), 
      reward = c("L_reward", "R_reward"),
      action = "Sub_Choose"
    ),
    behrule = list(
      cue = c("A", "B", "C", "D"),
      rsp = c("A", "B", "C", "D")
    ),
    generating = list(TD),
    candidates = list(TD),
    control = list(
      sample = 2,
      iter = 5
    )
  )

  rep_rec <- rpl_e(rec, option = list(plot = TRUE))
  testthat::expect_s3_class(rep_rec, "multiRLcpp_rpl_e")

  # Test S3 plot method
  testthat::expect_no_error(plot(rep_rec))
})
