# %%
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

