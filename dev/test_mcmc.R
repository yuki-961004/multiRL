library(multiRLcpp)
subject_one <- binaryRL::Mason_2024_G2[
  binaryRL::Mason_2024_G2[, "Subject"] == 1,
]
result <- estimate_mcmc(
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
    warmup = 50L,
    samples = 50L,
    chains = 2L,
    thin = 1L,
    step_size = 0.05,
    target_accept = 0.80,
    max_tree_depth = 4L,
    seed = 1004L
  )
)
summary(result)