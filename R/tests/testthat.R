library(testthat)
library(multiRLcpp)

# Prepend installed libs/x64 to PATH so LoadLibrary can find dependency DLLs (like torch_cpu.dll)
lib_dir <- system.file("libs/x64", package = "multiRLcpp")
if (nzchar(lib_dir)) {
  Sys.setenv(PATH = paste0(normalizePath(lib_dir), ";", Sys.getenv("PATH")))
}

test_check("multiRLcpp")
