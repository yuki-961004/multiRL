if (!requireNamespace("testthat", quietly = TRUE)) {
  stop("The testthat package is required.", call. = FALSE)
}

if (!requireNamespace("devtools", quietly = TRUE)) {
  stop("The devtools package is required.", call. = FALSE)
}

devtools::load_all("R", quiet = TRUE)
testthat::test_dir("tests/testthat")
