RSTD <- function(
    params = NULL,
    priors = NULL,
    lower = NULL,
    upper = NULL,
    settings = NULL
) {
  .model_free_builtin(
    name = "RSTD",
    free = list(
      alphaN = 0.3,
      alphaP = 0.3,
      beta = 0.5
    ),
    params = params,
    priors = priors,
    lower = lower,
    upper = upper,
    settings = settings
  )
}
