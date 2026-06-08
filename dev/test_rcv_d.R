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
  n_draws = 5L,
  seed = 1004L,
  threads = 32L,
  algorithm = "GN_MLSL",
  local_algorithm = "LN_BOBYQA",
  maxeval = 3L
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
  n_draws = 5L,
  seed = 1004L,
  threads = 32L,
  algorithm = "GN_MLSL",
  local_algorithm = "LN_BOBYQA",
  maxeval = 3L,
  maxiter = 2L
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
  n_draws = 10L,
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
  n_draws = 5L,
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
