# %%
data <- binaryRL::Mason_2024_G2

models <- list(
  multiRLcpp::TD,
  multiRLcpp::RSTD,
  multiRLcpp::Utility
)

settings <- list(
  list(name = "TD"),
  list(name = "RSTD"),
  list(name = "Utility")
)

mle <- multiRLcpp::rcv_d(
  estimator = "mle",
  data = data,
  id = 1,
  behrule = list(
    cue = c("A", "B", "C", "D"),
    rsp = c("A", "B", "C", "D")
  ),
  colnames = list(
    object = c("L_choice", "R_choice"),
    reward = c("L_reward", "R_reward"),
    action = "Sub_Choose"
  ),
  models = models,
  settings = settings,
  lowers = list(c(0, 0), c(0, 0, 0), c(0, 0, 0)),
  uppers = list(c(1, 5), c(1, 1, 5), c(1, 5, 1)),
  control = list(
    n_draws = 1,
    seed = 1004,
    threads = 32
  ),
  fit_control = list(
    algorithm = "LN_BOBYQA",
    local_algorithm = "LN_BOBYQA",
    maxeval = 10,
    seed = 1004
  )
)

base::stopifnot(base::inherits(mle, "multiRLcpp_rcv_d"))
base::stopifnot(base::nrow(mle$simulation) > 0L)
base::stopifnot(base::nrow(mle$truth) == 3L)
base::stopifnot(base::nrow(mle$model_recovery) == 9L)
base::stopifnot(base::all(
  c("TD", "RSTD", "Utility") %in% mle$truth$generating_model
))

print(mle$recovery)
print(mle$model_recovery)

# %%
data <- binaryRL::Mason_2024_G2

models <- list(
  multiRLcpp::TD,
  multiRLcpp::RSTD,
  multiRLcpp::Utility
)

settings <- list(
  list(name = "TD"),
  list(name = "RSTD"),
  list(name = "Utility")
)

abc <- multiRLcpp::rcv_d(
  estimator = "abc",
  data = data,
  id = 1,
  behrule = list(
    cue = c("A", "B", "C", "D"),
    rsp = c("A", "B", "C", "D")
  ),
  colnames = list(
    object = c("L_choice", "R_choice"),
    reward = c("L_reward", "R_reward"),
    action = "Sub_Choose"
  ),
  models = models,
  settings = settings,
  lowers = list(c(0, 0), c(0, 0, 0), c(0, 0, 0)),
  uppers = list(c(1, 5), c(1, 1, 5), c(1, 5, 1)),
  control = list(
    n_draws = 1,
    seed = 1004,
    threads = 32,
    scope = "shared"
  ),
  fit_control = list(
    samples = 10,
    tol = 0.5,
    method = "rejection",
    reduction = "none",
    threads = 1,
    print_level = 0
  )
)

base::stopifnot(base::inherits(abc, "multiRLcpp_rcv_d"))
base::stopifnot(base::nrow(abc$simulation) > 0L)
base::stopifnot(base::nrow(abc$truth) == 3L)
base::stopifnot(base::nrow(abc$model_recovery) == 9L)
base::stopifnot(abc$estimator$scope == "shared")

print(abc$recovery)
print(abc$model_recovery)
