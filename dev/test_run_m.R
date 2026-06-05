library(multiRLcpp)

data <- binaryRL::Mason_2024_G2[
  binaryRL::Mason_2024_G2[, "Subject"] == 1,
]

result <- run_m(
  data = data,
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

base::print(result$fit)
base::stopifnot(base::round(result$fit$LogL, 2) == -317.45)
base::cat("run_m smoke test passed.\n")
