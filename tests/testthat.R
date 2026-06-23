if (!requireNamespace("testthat", quietly = TRUE)) {
  stop("The testthat package is required.", call. = FALSE)
}

if (!requireNamespace("devtools", quietly = TRUE)) {
  stop("The devtools package is required.", call. = FALSE)
}

# Prepend R/src to PATH so LoadLibrary can find dependency DLLs (like torch_cpu.dll)
src_dir <- base::normalizePath("R/src", winslash = "/", mustWork = FALSE)
if (base::dir.exists(src_dir)) {
  base::Sys.setenv(PATH = base::paste0(src_dir, ";", base::Sys.getenv("PATH")))
}

devtools::load_all("R", quiet = TRUE)
testthat::test_dir("tests/testthat")
