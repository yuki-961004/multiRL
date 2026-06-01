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

# %%
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

base::summary(binaryRL.res)

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
    free = list(
      alpha = 0.3,
      beta = 0.5
    ),
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

multiRL.summary <- multiRL::summary(multiRL.model)

multiRLcpp.model <- run_m(
  engine = "Cpp",
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
    free = list(
      alpha = 0.3,
      beta = 0.5
    ),
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

base::summary(multiRLcpp.model)

# %%

multiRLcpp.mle <- estimate_mle(
  engine = "Cpp",
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
    free = list(
      alpha = 0.3,
      beta = 0.5
    ),
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
    maxeval = 100,
    xtol_rel = 1e-8
  )
)

base::summary(multiRLcpp.mle)

multiRL.mle <- multiRL::fit_p(
  estimate = "MLE",
  data = subject_one,
  ids = 1,
  behrule = list(
    cue = c("A", "B", "C", "D"),
    rsp = c("A", "B", "C", "D")
  ),
  colnames = list(
    object = c("L_choice", "R_choice"),
    reward = c("L_reward", "R_reward"),
    action = "Sub_Choose"
  ),
  models = list(multiRL::TD),
  settings = list(
    list(
      name = "TD",
      policy = "off",
      estimate = "MLE"
    )
  ),
  priors = list(
    list(
      alpha = function(x) {
        stats::rbeta(n = 1, shape1 = 2, shape2 = 2)
      },
      beta = function(x) {
        stats::rexp(n = 1, rate = 1)
      }
    )
  ),
  lowers = list(c(0, 0)),
  uppers = list(c(1, 1)),
  control = list(
    core = 1,
    iter = 20,
    seed = 123,
    algorithm = "NLOPT_LN_BOBYQA"
  )
)

multiRL.mle.td <- multiRL.mle[[1L]]
multiRL.mle.params <- c(
  alpha = multiRL.mle.td$alpha[[1L]],
  beta = multiRL.mle.td$beta[[1L]]
)
multiRLcpp.mle.params <- c(
  alpha = multiRLcpp.mle$params[["alpha"]],
  beta = multiRLcpp.mle$params[["beta"]]
)

base::print(
  base::rbind(
    multiRL = multiRL.mle.params,
    multiRLcpp = multiRLcpp.mle.params
  )
)

# %%
benchmark <- base::rbind(
  binaryRL = extract_binary_metric(binaryRL.res),
  multiRL = extract_multiRL_metric(multiRL.model),
  multiRLcpp = extract_multiRLcpp_metric(multiRLcpp.model)
)

base::print(benchmark)

logL <- benchmark[, "LogL"]
logL_equal <- base::all(base::abs(logL - logL[["binaryRL"]]) < 1e-4)

base::cat(
  "Rounded LogL:",
  base::paste(base::round(logL, 2), collapse = ", "),
  "\n"
)

if (!logL_equal) {
  base::stop("LogL values are not equal within tolerance.")
}

params_equal <- base::all(
  base::abs(multiRL.mle.params - multiRLcpp.mle.params) < 1e-4
)

if (!params_equal) {
  base::warning(
    "multiRL and multiRLcpp estimate_mle parameters do not match."
  )
}

# %%
run_m_iterations <- 100L

time_quietly <- function(code) {
  output_connection <- base::file(base::nullfile(), open = "wt")
  message_connection <- base::file(base::nullfile(), open = "wt")

  base::sink(file = output_connection)
  base::sink(file = message_connection, type = "message")

  base::on.exit({
    if (base::sink.number(type = "message") > 0L) {
      base::sink(type = "message")
    }

    if (base::sink.number(type = "output") > 0L) {
      base::sink(type = "output")
    }

    base::close(message_connection)
    base::close(output_connection)
  })

  base::system.time(base::force(code))
}

run_multiRL_speed_once <- function() {
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

  base::invisible(NULL)
}

run_multiRLcpp_speed_once <- function() {
  run_m(
    engine = "Cpp",
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

  base::invisible(NULL)
}

multiRL_time <- time_quietly({
  for (iteration in base::seq_len(run_m_iterations)) {
    run_multiRL_speed_once()
  }
})

multiRLcpp_time <- time_quietly({
  for (iteration in base::seq_len(run_m_iterations)) {
    run_multiRLcpp_speed_once()
  }
})

run_m_speed <- base::data.frame(
  engine = c("multiRL", "multiRLcpp"),
  iterations = c(run_m_iterations, run_m_iterations),
  elapsed = c(
    base::unname(multiRL_time[["elapsed"]]),
    base::unname(multiRLcpp_time[["elapsed"]])
  )
)

run_m_speed$per_iteration <- run_m_speed$elapsed / run_m_speed$iterations
run_m_speed$speedup <- run_m_speed$elapsed[run_m_speed$engine == "multiRL"] /
  run_m_speed$elapsed

base::print(run_m_speed)
