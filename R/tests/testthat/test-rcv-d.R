testthat::local_edition(3)

testthat::skip_if_not_installed("multiRL")

# Pre-extract data for speed
data_sub <- multiRL::TAB[multiRL::TAB[, "Subject"] == 1, ]

testthat::test_that("MLE recovery diagnostics runs and returns correct class", {
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
  testthat::expect_s3_class(rec, "multiRLcpp_rcv_d")
})
