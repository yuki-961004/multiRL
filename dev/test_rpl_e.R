
# %%

data <- binaryRL::Mason_2024_G2

result.fit <- fit_p(
  data = data,
  estimator = "mle",
  id = c(1:10),
  behrule = list(
    cue = c("A", "B", "C", "D"),
    rsp = c("A", "B", "C", "D")
  ),
  colnames = list(
    object = c("L_choice", "R_choice"),
    reward = c("L_reward", "R_reward"),
    action = "Sub_Choose"
  ),
  models = list(TD, Utility),
  control = list(
    algorithm = "LN_BOBYQA",
    local_algorithm = "LN_BOBYQA",
    maxeval = 10L,
    seed = 1004L
  )
)

base::stopifnot(base::is.list(result.fit))
base::stopifnot("model" %in% base::names(result.fit$fit))
base::stopifnot(base::length(base::unique(result.fit$fit$model)) == 2L)

result.rpl <- rpl_e(
  result = result.fit,
  option = list(plot = TRUE)
)

base::stopifnot(base::is.list(result.rpl))
base::stopifnot(base::all(c(
  "input",
  "replay",
  "plot_data",
  "diagnostics"
) %in% base::names(result.rpl)))
base::stopifnot("Human" %in% result.rpl$plot_data$model)
base::stopifnot(base::any(result.rpl$plot_data$source == "model"))
base::stopifnot(result.rpl$diagnostics$policy == "on")

base::print(result.fit$fit)
base::print(utils::head(result.rpl$plot_data))
base::cat("rpl_e smoke test passed.\n")

# %%
# rcv_d + rpl_e recovery plotting

data <- binaryRL::Mason_2024_G2

models <- list(
  multiRLcpp::TD,
  multiRLcpp::RSTD,
  multiRLcpp::Utility
)

settings <- list(
  list(name = "TD", policy = "off"),
  list(name = "RSTD", policy = "off"),
  list(name = "Utility", policy = "off")
)

result.rcv <- multiRLcpp::rcv_d(
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
    n_draws = 10L,
    threads = 32L,
    algorithm = "GN_MLSL",
    local_algorithm = "LN_BOBYQA",
    maxeval = 10L,
    seed = 1004L
  )
)

base::stopifnot(base::inherits(result.rcv, "multiRLcpp_rcv_d"))
base::stopifnot(base::nrow(result.rcv$recovery) > 0L)
base::stopifnot(base::nrow(result.rcv$model_recovery) > 0L)

result.rpl.rcv <- rpl_e(
  result = result.rcv,
  option = list(plot = TRUE)
)

base::stopifnot(base::is.list(result.rpl.rcv))
base::stopifnot("recovery" %in% base::names(result.rpl.rcv))
base::stopifnot("model_recovery" %in% base::names(result.rpl.rcv))
base::stopifnot("plot" %in% base::names(result.rpl.rcv))
base::stopifnot(!base::is.null(result.rpl.rcv$plot))

base::print(utils::head(result.rcv$recovery))
base::cat("rpl_e rcv_d recovery test passed.\n")
