.shell_standardize_priors <- function(priors, params) {
  if (base::length(priors) == 0L) {
    return(.shell_empty_priors())
  }

  if (!base::identical(base::names(priors), base::names(params))) {
    base::stop(
      "The names of 'priors' must be identical to the names of 'params'."
    )
  }

  specs <- base::lapply(priors, .shell_prior_to_spec)

  list(
    active = TRUE,
    name = base::names(specs),
    type = base::vapply(specs, `[[`, character(1L), "type"),
    param1 = base::vapply(specs, `[[`, numeric(1L), "param1"),
    param2 = base::vapply(specs, `[[`, numeric(1L), "param2")
  )
}

.shell_empty_priors <- function() {
  list(
    active = FALSE,
    name = character(),
    type = character(),
    param1 = numeric(),
    param2 = numeric()
  )
}

.shell_prior_to_spec <- function(prior) {
  if (base::is.list(prior) && !base::is.null(prior$type)) {
    return(.shell_prior_list_to_spec(prior))
  }

  base::stop(
    "Priors must be structured prior lists."
  )
}

.shell_prior_list_to_spec <- function(prior) {
  type <- base::tolower(base::as.character(prior$type[[1L]]))

  list(
    type = .shell_normalize_prior_type(type),
    param1 = .shell_prior_arg1(prior, type),
    param2 = .shell_prior_arg2(prior, type)
  )
}

.shell_normalize_prior_type <- function(type) {
  switch(
    type,
    norm = "normal",
    normal = "normal",
    unif = "uniform",
    uniform = "uniform",
    lnorm = "lognormal",
    lognormal = "lognormal",
    cauchy = "cauchy",
    beta = "beta",
    exp = "exponential",
    exponential = "exponential",
    none = "none",
    base::stop("Unsupported prior type.")
  )
}

.shell_prior_arg1 <- function(prior, type) {
  switch(
    .shell_normalize_prior_type(type),
    normal = .shell_named_number(prior, c("mean", "mu", "param1")),
    uniform = .shell_named_number(prior, c("min", "lower", "param1")),
    lognormal = .shell_named_number(prior, c("meanlog", "mean", "param1")),
    cauchy = .shell_named_number(prior, c("location", "mean", "param1")),
    beta = .shell_named_number(prior, c("shape1", "alpha", "param1")),
    exponential = .shell_named_number(prior, c("rate", "lambda", "param1")),
    none = NaN
  )
}

.shell_prior_arg2 <- function(prior, type) {
  switch(
    .shell_normalize_prior_type(type),
    normal = .shell_named_number(prior, c("sd", "sigma", "param2")),
    uniform = .shell_named_number(prior, c("max", "upper", "param2")),
    lognormal = .shell_named_number(prior, c("sdlog", "sd", "param2")),
    cauchy = .shell_named_number(prior, c("scale", "sd", "param2")),
    beta = .shell_named_number(prior, c("shape2", "beta", "param2")),
    exponential = NaN,
    none = NaN
  )
}

.shell_named_number <- function(x, keys) {
  for (key in keys) {
    if (!base::is.null(x[[key]])) {
      return(base::as.numeric(x[[key]][[1L]]))
    }
  }

  base::stop("Structured prior list is missing a required argument.")
}
