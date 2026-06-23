testthat::local_edition(3)

testthat::test_that("TD model initialization and customization", {
  m <- TD()
  testthat::expect_type(m, "list")
  testthat::expect_equal(m$model, "TD")
  testthat::expect_equal(m$process, "process_model_free")
  testthat::expect_equal(m$params$free$alpha, 0.3)
  testthat::expect_equal(m$params$free$beta, 0.5)

  m2 <- TD(params = list(free = list(alpha = 0.6)))
  testthat::expect_equal(m2$params$free$alpha, 0.6)
})

testthat::test_that("RSTD model initialization", {
  m <- RSTD()
  testthat::expect_type(m, "list")
  testthat::expect_equal(m$model, "RSTD")
  testthat::expect_equal(m$params$free$alphaN, 0.3)
  testthat::expect_equal(m$params$free$alphaP, 0.3)
})

testthat::test_that("Utility model initialization", {
  m <- Utility()
  testthat::expect_type(m, "list")
  testthat::expect_equal(m$model, "Utility")
  testthat::expect_equal(m$params$free$gamma, 0.5)
})
