# Optional dependency install from the repository root:
# $env:MULTIRL_ENABLE_RNN = "ON"
# $env:MULTIRL_ENABLE_MCMC = "ON"
# $env:MULTIRL_LIBTORCH_DIR = "./build/_deps/libtorch-src"
#
# cmake -S . -B build
# cmake --build build --config Release
# Windows Rtools/GCC cannot compile the official Windows LibTorch package.
# Use the root CMake MSVC build or Python package for RNN on Windows.

# %%
Sys.setenv(MULTIRL_ENABLE_RNN = "OFF")   # 必须为 OFF，不让 Rtools 去链接 LibTorch
Sys.setenv(MULTIRL_ENABLE_MCMC = "ON")   # 开启 MCMC 编译支持
# 1. 使用 normalizePath 自动将斜杠转换为 Windows 规范的反斜杠
libtorch_path <- normalizePath("E:/YuKi_Project/Software/RL/multiRL-remake/build/_deps/libtorch-src/lib", mustWork = TRUE)
libs_x64_path <- normalizePath("E:/YuKi_Project/Software/RL/multiRL-remake/R/inst/libs/x64", mustWork = TRUE)
# 2. 将它们加入 PATH
Sys.setenv(PATH = paste(libs_x64_path, libtorch_path, Sys.getenv("PATH"), sep = ";"))
#devtools::clean_dll("./R")
devtools::load_all("./R")

# %%
# Common setup for all rcv_d reproducibility tests

data <- binaryRL::Mason_2024_G2

behrule <- list(
  cue = c("A", "B", "C", "D"),
  rsp = c("A", "B", "C", "D")
)

colnames <- list(
  object = c("L_choice", "R_choice"),
  reward = c("L_reward", "R_reward"),
  action = "Sub_Choose"
)

models <- list(multiRLcpp::TD)
settings <- list(list(name = "TD"))
lowers <- list(c(0, 0))
uppers <- list(c(1, 5))

# %%
# rcv_d reproducibility test: MLE
# Tests that rcv_d with estimator = "mle" produces identical results
# when run twice with the same seed.

control_mle <- list(
  n_draws = 50L,
  seed = 1004L,
  threads = 32L,
  algorithm = "GN_MLSL",
  local_algorithm = "LN_BOBYQA",
  maxeval = 10L
)

mle_1 <- multiRLcpp::rcv_d(
  estimator = "mle",
  data = data, id = 1,
  behrule = behrule, colnames = colnames,
  models = models, settings = settings,
  lowers = lowers, uppers = uppers,
  control = control_mle
)

mle_2 <- multiRLcpp::rcv_d(
  estimator = "mle",
  data = data, id = 1,
  behrule = behrule, colnames = colnames,
  models = models, settings = settings,
  lowers = lowers, uppers = uppers,
  control = control_mle
)

recovery_1 <- mle_1$recovery
recovery_2 <- mle_2$recovery
recovery_match <- base::all(
  base::abs(recovery_1$true - recovery_2$true) < 1e-10
)
recovered_match <- base::all(
  base::abs(recovery_1$recovered - recovery_2$recovered) < 1e-6
)

base::cat("MLE rcv_d reproducibility test:\n")
base::cat("  true values match:", recovery_match, "\n")
base::cat("  recovered values match:", recovered_match, "\n")
base::stopifnot(recovery_match, recovered_match)
base::cat("MLE rcv_d reproducibility test PASSED.\n")

# %%
# rcv_d reproducibility test: MAP

control_map <- list(
  n_draws = 50L,
  seed = 1004L,
  threads = 32L,
  algorithm = "GN_MLSL",
  local_algorithm = "LN_BOBYQA",
  maxeval = 10L,
  maxiter = 10L
)

map_1 <- multiRLcpp::rcv_d(
  estimator = "map",
  data = data, id = 1,
  behrule = behrule, colnames = colnames,
  models = models, settings = settings,
  lowers = lowers, uppers = uppers,
  control = control_map
)

map_2 <- multiRLcpp::rcv_d(
  estimator = "map",
  data = data, id = 1,
  behrule = behrule, colnames = colnames,
  models = models, settings = settings,
  lowers = lowers, uppers = uppers,
  control = control_map
)

recovery_1 <- map_1$recovery
recovery_2 <- map_2$recovery
recovery_match <- base::all(
  base::abs(recovery_1$true - recovery_2$true) < 1e-10
)
recovered_match <- base::all(
  base::abs(recovery_1$recovered - recovery_2$recovered) < 1e-6
)

base::cat("MAP rcv_d reproducibility test:\n")
base::cat("  true values match:", recovery_match, "\n")
base::cat("  recovered values match:", recovered_match, "\n")
base::stopifnot(recovery_match, recovered_match)
base::cat("MAP rcv_d reproducibility test PASSED.\n")

# %%
# rcv_d reproducibility test: ABC

control_abc <- list(
  n_draws = 50L,
  seed = 1004L,
  threads = 32L
)

abc_1 <- multiRLcpp::rcv_d(
  estimator = "abc",
  data = data, id = 1,
  behrule = behrule, colnames = colnames,
  models = models, settings = settings,
  lowers = lowers, uppers = uppers,
  control = control_abc
)

abc_2 <- multiRLcpp::rcv_d(
  estimator = "abc",
  data = data, id = 1,
  behrule = behrule, colnames = colnames,
  models = models, settings = settings,
  lowers = lowers, uppers = uppers,
  control = control_abc
)

recovery_1 <- abc_1$recovery
recovery_2 <- abc_2$recovery
recovery_match <- base::all(
  base::abs(recovery_1$true - recovery_2$true) < 1e-10
)
recovered_match <- base::all(
  base::abs(recovery_1$recovered - recovery_2$recovered) < 1e-6
)

base::cat("ABC rcv_d reproducibility test:\n")
base::cat("  true values match:", recovery_match, "\n")
base::cat("  recovered values match:", recovered_match, "\n")
base::stopifnot(recovery_match, recovered_match)
base::cat("ABC rcv_d reproducibility test PASSED.\n")

# %%
# rcv_d reproducibility test: MCMC

control_mcmc <- list(
  n_draws = 50L,
  seed = 1004L,
  threads = 32L,
  chains = 2L,
  samples = 10L,
  warmup = 5L
)

mcmc_1 <- multiRLcpp::rcv_d(
  estimator = "mcmc",
  data = data, id = 1,
  behrule = behrule, colnames = colnames,
  models = models, settings = settings,
  lowers = lowers, uppers = uppers,
  control = control_mcmc
)

mcmc_2 <- multiRLcpp::rcv_d(
  estimator = "mcmc",
  data = data, id = 1,
  behrule = behrule, colnames = colnames,
  models = models, settings = settings,
  lowers = lowers, uppers = uppers,
  control = control_mcmc
)

recovery_1 <- mcmc_1$recovery
recovery_2 <- mcmc_2$recovery
recovery_match <- base::all(
  base::abs(recovery_1$true - recovery_2$true) < 1e-10
)
recovered_match <- base::all(
  base::abs(recovery_1$recovered - recovery_2$recovered) < 1e-6
)

base::cat("MCMC rcv_d reproducibility test:\n")
base::cat("  true values match:", recovery_match, "\n")
base::cat("  recovered values match:", recovered_match, "\n")
base::stopifnot(recovery_match, recovered_match)
base::cat("MCMC rcv_d reproducibility test PASSED.\n")

# %%
# rcv_d device test: RNN (LibTorch CPU vs GPU)
# The same seed is used for both devices. CPU and GPU floating-point
# kernels can still differ slightly, so this chunk reports the difference
# instead of requiring bit-identical recovery estimates.

control_rnn_cpu <- list(
  n_draws = 50L,
  epochs = 3L,
  batch_size = 32L,
  units = 32L,
  layers = 1L,
  dropout = 0,
  learning_rate = 0.001,
  seed = 1004L,
  threads = 32L,
  layer = "gru",
  verbose = 0L,
  device = "cpu"
)

control_rnn_gpu <- utils::modifyList(
  x = control_rnn_cpu,
  val = list(
    threads = 0L,
    device = "gpu"
  )
)

tryCatch({
  rnn_cpu <- multiRLcpp::rcv_d(
    estimator = "rnn",
    data = data, id = 1,
    behrule = behrule, colnames = colnames,
    models = models, settings = settings,
    lowers = lowers, uppers = uppers,
    control = control_rnn_cpu
  )

  rnn_gpu <- multiRLcpp::rcv_d(
    estimator = "rnn",
    data = data, id = 1,
    behrule = behrule, colnames = colnames,
    models = models, settings = settings,
    lowers = lowers, uppers = uppers,
    control = control_rnn_gpu
  )

  recovery_1 <- rnn_cpu$recovery
  recovery_2 <- rnn_gpu$recovery
  recovery_match <- base::all(
    base::abs(recovery_1$true - recovery_2$true) < 1e-10
  )
  recovered_diff <- base::max(
    base::abs(recovery_1$recovered - recovery_2$recovered),
    na.rm = TRUE
  )
  recovered_finite <- base::all(base::is.finite(recovery_1$recovered)) &&
    base::all(base::is.finite(recovery_2$recovered))

  base::cat("RNN rcv_d CPU/GPU device test:\n")
  base::cat("  true values match:", recovery_match, "\n")
  base::cat("  max recovered difference:", recovered_diff, "\n")
  base::cat("  recovered values are finite:", recovered_finite, "\n")
  base::stopifnot(recovery_match, recovered_finite)
  base::cat("RNN rcv_d CPU/GPU device test PASSED.\n")
}, error = function(e) {
  if (grepl("requires LibTorch support", e$message)) {
    base::cat("Skipping RNN CPU/GPU device tests because RNN backend is not enabled in this build.\n")
  } else {
    stop(e)
  }
})
