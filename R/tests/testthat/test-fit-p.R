testthat::local_edition(3)

testthat::skip_if_not_installed("multiRL")

# Pre-extract data for speed
data_sub <- multiRL::TAB[multiRL::TAB[, "Subject"] %in% 1:2, ]

testthat::test_that("MLE fitting runs and returns correct class", {
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
    models = list(TD, RSTD),
    control = list(iter = 5)
  )
  testthat::expect_s3_class(fit, "multiRLcpp_fit_p")
  testthat::expect_equal(fit$diagnostics$n_models, 2)
})

testthat::test_that("MAP fitting runs and returns correct class", {
  fit <- fit_p(
    estimator = "map",
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
    control = list(iter = c(5, 1))
  )
  testthat::expect_s3_class(fit, "multiRLcpp_fit_p")
})

testthat::test_that("ABC fitting runs with PLS/PCA/None reduction", {
  for (red in list(NULL, "PLS", "PCA")) {
    fit <- fit_p(
      estimator = "abc",
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
      control = list(
        sample = 10,
        train = 10,
        tol = 0.5,
        reduction = red
      )
    )
    testthat::expect_s3_class(fit, "multiRLcpp_fit_p")
  }
})

testthat::test_that("MCMC fitting skip or success", {
  res <- tryCatch({
    fit_p(
      estimator = "mcmc",
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
      control = list(
        warmup = 2,
        samples = 3,
        chains = 1
      )
    )
  }, error = function(e) {
    if (base::grepl("Stan Math", e$message) || base::grepl("MCMC is not available", e$message)) {
      return("skipped")
    } else {
      stop(e)
    }
  })

  if (base::identical(res, "skipped")) {
    testthat::skip("Stan Math support not compiled")
  } else {
    testthat::expect_s3_class(res, "multiRLcpp_fit_p")
  }
})

testthat::test_that("RNN fitting skip or success", {
  res <- tryCatch({
    fit_p(
      estimator = "rnn",
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
      control = list(
        epochs = 1,
        n_draws = 10,
        batch_size = 5
      )
    )
  }, error = function(e) {
    if (base::grepl("LibTorch", e$message) || base::grepl("multiRL_torch_backend", e$message)) {
      return("skipped")
    } else {
      stop(e)
    }
  })

  if (base::identical(res, "skipped")) {
    testthat::skip("LibTorch support not compiled")
  } else {
    testthat::expect_s3_class(res, "multiRLcpp_fit_p")
  }
})
