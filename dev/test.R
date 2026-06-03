# %%
devtools::clean_dll("./R")
devtools::load_all("./R")


# %%
extract_binary_metric <- function(model) {
  utils::capture.output(summary_value <- base::summary(model))
  metric <- summary_value[[2]]
  c(
    LogL = metric[metric[, "Metric"] == "LogL", "Value"],
    LogPr = metric[metric[, "Metric"] == "LogPr", "Value"],
    LogPo = metric[metric[, "Metric"] == "LogPo", "Value"]
  )
}

extract_multiRL_metric <- function(model) {
  utils::capture.output(summary_value <- multiRL::summary(model))
  c(
    LogL = summary_value@metrics@LL,
    LogPr = summary_value@metrics@LPr,
    LogPo = summary_value@metrics@LPo
  )
}

extract_multiRLcpp_metric <- function(model) {
  c(
    LogL = model$fit$LogL,
    LogPr = model$fit$LogPr,
    LogPo = model$fit$LogPo
  )
}

subject_one <- binaryRL::Mason_2024_G2[
  binaryRL::Mason_2024_G2[, "Subject"] == 1,
]

binaryRL.res <- binaryRL::run_m(
  mode = "fit",
  data = binaryRL::Mason_2024_G2,
  id = 1,
  n_params = 3,
  n_trials = 360,
  pi = 0.1,
  eta = c(0.3),
  tau = c(0.5),
  threshold = 20,
  priors = list(
    eta = function(x) {
      stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)
    },
    tau = function(x) {
      stats::dexp(x, rate = 1, log = TRUE)
    }
  ),
  policy = "off"
)

multiRL.model <- multiRL::run_m(
  engine = "R",
  data = subject_one,
  behrule = list(
    cue = c("A", "B", "C", "D"),
    rsp = c("A", "B", "C", "D")
  ),
  colnames = list(
    object = c("L_choice", "R_choice"),
    reward = c("L_reward", "R_reward"),
    action = "Sub_Choose"
  ),
  params = list(
    free = list(alpha = 0.3, beta = 0.5),
    fixed = list(threshold = 20)
  ),
  priors = list(
    alpha = function(x) {
      stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)
    },
    beta = function(x) {
      stats::dexp(x, rate = 1, log = TRUE)
    }
  ),
  settings = list(
    name = "TD",
    mode = "fitting",
    estimate = "MLE",
    policy = "off"
  )
)

multiRLcpp.model <- run_m(
  data = subject_one,
  behrule = list(
    cue = c("A", "B", "C", "D"),
    rsp = c("A", "B", "C", "D")
  ),
  colnames = list(
    object = c("L_choice", "R_choice"),
    reward = c("L_reward", "R_reward"),
    action = "Sub_Choose"
  ),
  params = list(
    free = list(alpha = 0.3, beta = 0.5),
    fixed = list(threshold = 20)
  ),
  priors = list(
    alpha = list(type = "beta", shape1 = 2, shape2 = 2),
    beta = list(type = "exp", rate = 1)
  ),
  settings = list(
    name = "TD",
    mode = "fitting",
    estimate = "MLE",
    policy = "off"
  )
)

benchmark <- base::rbind(
  binaryRL = extract_binary_metric(binaryRL.res),
  multiRL = extract_multiRL_metric(multiRL.model),
  multiRLcpp = extract_multiRLcpp_metric(multiRLcpp.model)
)

base::print(benchmark)

logL <- benchmark[, "LogL"]
if (!base::all(base::abs(logL - logL[["binaryRL"]]) < 1e-4)) {
  base::stop("LogL values are not equal within tolerance.")
}

base::cat(
  "Rounded LogL:",
  base::paste(base::round(logL, 2), collapse = ", "),
  "\n"
)

run_m_iterations <- 100L

multiRL_time <- base::system.time({
  for (iteration in base::seq_len(run_m_iterations)) {
    multiRL::run_m(
      engine = "R",
      data = subject_one,
      behrule = list(
        cue = c("A", "B", "C", "D"),
        rsp = c("A", "B", "C", "D")
      ),
      colnames = list(
        object = c("L_choice", "R_choice"),
        reward = c("L_reward", "R_reward"),
        action = "Sub_Choose"
      ),
      params = list(
        free = list(alpha = 0.3, beta = 0.5),
        fixed = list(threshold = 20)
      ),
      settings = list(
        name = "TD",
        mode = "fitting",
        estimate = "MLE",
        policy = "off"
      )
    )
  }
})

multiRLcpp_time <- base::system.time({
  for (iteration in base::seq_len(run_m_iterations)) {
    run_m(
      data = subject_one,
      behrule = list(
        cue = c("A", "B", "C", "D"),
        rsp = c("A", "B", "C", "D")
      ),
      colnames = list(
        object = c("L_choice", "R_choice"),
        reward = c("L_reward", "R_reward"),
        action = "Sub_Choose"
      ),
      params = list(
        free = list(alpha = 0.3, beta = 0.5),
        fixed = list(threshold = 20)
      ),
      settings = list(
        name = "TD",
        mode = "fitting",
        estimate = "MLE",
        policy = "off"
      )
    )
  }
})

base::print(
  base::data.frame(
    package = c("multiRL", "multiRLcpp"),
    iterations = c(run_m_iterations, run_m_iterations),
    elapsed = c(
      base::unname(multiRL_time[["elapsed"]]),
      base::unname(multiRLcpp_time[["elapsed"]])
    )
  )
)

# %%

multiRLcpp.mle <- estimate_mle(
  data = binaryRL::Mason_2024_G2,
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
  params = list(
    free = list(alpha = 0.3, beta = 0.5),
    fixed = list(threshold = 20)
  ),
  lower = c(0, 0),
  upper = c(1, 1),
  settings = list(
    name = "TD",
    mode = "fitting",
    estimate = "MLE",
    policy = "off"
  ),
  control = list(
    algorithm = "LN_BOBYQA",
    local_algorithm = "LN_BOBYQA",
    maxeval = 100,
    xtol_rel = 1e-8
  )
)

base::summary(multiRLcpp.mle)

# %%

multiRLcpp.map <- estimate_map(
  data = binaryRL::Mason_2024_G2,
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
  params = list(
    free = list(alpha = 0.3, beta = 0.5),
    fixed = list(threshold = 20)
  ),
  priors = list(
    alpha = list(type = "beta", shape1 = 2, shape2 = 2),
    beta = list(type = "exp", rate = 1)
  ),
  lower = c(0, 0),
  upper = c(1, 1),
  settings = list(
    name = "TD",
    mode = "fitting",
    estimate = "MAP",
    policy = "off"
  ),
  control = list(
    algorithm = "LN_BOBYQA",
    local_algorithm = "LN_BOBYQA",
    mle_maxeval = 100,
    map_maxiter = 10,
    map_tol = 1e-3,
    map_patience = 10,
    xtol_rel = 1e-8
  )
)

base::summary(multiRLcpp.map)

# %%

multiRLcpp.mcmc <- estimate_mcmc(
  data = binaryRL::Mason_2024_G2,
  id = c(1:2),
  behrule = list(
    cue = c("A", "B", "C", "D"),
    rsp = c("A", "B", "C", "D")
  ),
  colnames = list(
    object = c("L_choice", "R_choice"),
    reward = c("L_reward", "R_reward"),
    action = "Sub_Choose"
  ),
  params = list(
    free = list(alpha = 0.3, beta = 0.5),
    fixed = list(threshold = 20)
  ),
  priors = list(
    alpha = list(type = "beta", shape1 = 2, shape2 = 2),
    beta = list(type = "exp", rate = 1)
  ),
  lower = c(0, 0),
  upper = c(1, 1),
  settings = list(
    name = "TD",
    mode = "fitting",
    estimate = "MCMC",
    policy = "off"
  ),
  control = list(
    algorithm = "nuts",
    warmup = 25L,
    samples = 50L,
    chains = 2L,
    thin = 1L,
    step_size = 0.05,
    target_accept = 0.80,
    max_tree_depth = 4L,
    seed = 1004L
  )
)

base::summary(multiRLcpp.mcmc)

# %%

multiRLcpp.abc <- estimate_abc(
  data = binaryRL::Mason_2024_G2,
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
  params = list(
    free = list(alpha = 0.3, beta = 0.5),
    fixed = list(threshold = 20)
  ),
  lower = c(0, 0),
  upper = c(1, 1),
  settings = list(
    name = "TD",
    mode = "fitting",
    estimate = "ABC",
    policy = "off"
  ),
  control = list(
    samples = 1000L,
    tol = 0.2,
    method = "rejection",
    reduction = "pls",
    fake_block = 4L,
    seed = 1004L,
    threads = 1L,
    print_level = 0L
  )
)

print(base::names(multiRLcpp.abc))
print(multiRLcpp.abc$fit)
print(multiRLcpp.abc$diagnostics$subjects)
stopifnot(multiRLcpp.abc$estimator$name == "ABC")
stopifnot(multiRLcpp.abc$estimator$backend == "abcpp")
stopifnot(multiRLcpp.abc$diagnostics$subjects$fake_block[1L] == 4L)
stopifnot(multiRLcpp.abc$diagnostics$subjects$n_blocks_used[1L] == 4L)
stopifnot(
  multiRLcpp.abc$diagnostics$subjects$n_comp_used[1L] ==
    multiRLcpp.abc$diagnostics$subjects$n_blocks_used[1L]
)
