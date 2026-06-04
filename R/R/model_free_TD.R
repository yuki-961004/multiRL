TD <- function(
    params = NULL,
    priors = NULL,
    lower = NULL,
    upper = NULL,
    settings = NULL
) {
  .model_free_builtin(
    name = "TD",
    free = list(
      alpha = 0.3,
      beta = 0.5
    ),
    params = params,
    priors = priors,
    lower = lower,
    upper = upper,
    settings = settings
  )
}
