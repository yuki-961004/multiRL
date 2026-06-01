.modify_priors <- function(priors, params) {
  if (base::length(priors) == 0L) {
    return(.modify_priors_empty())
  }

  if (!base::identical(base::names(priors), base::names(params))) {
    base::stop(
      "The names of 'priors' must be identical to the names of 'params'."
    )
  }

  specs <- base::lapply(priors, .modify_priors_to_spec)

  list(
    active = TRUE,
    name = base::names(specs),
    type = base::vapply(specs, `[[`, character(1L), "type"),
    param1 = base::vapply(specs, `[[`, numeric(1L), "param1"),
    param2 = base::vapply(specs, `[[`, numeric(1L), "param2")
  )
}

.modify_priors_empty <- function() {
  list(
    active = FALSE,
    name = character(),
    type = character(),
    param1 = numeric(),
    param2 = numeric()
  )
}

.modify_priors_to_spec <- function(prior) {
  if (base::is.list(prior) && !base::is.null(prior$type)) {
    return(.modify_priors_list_to_spec(prior))
  }

  base::stop("Priors must be structured prior lists.")
}

.modify_priors_list_to_spec <- function(prior) {
  type <- base::tolower(base::as.character(prior$type[[1L]]))

  list(
    type = .modify_priors_normalize_type(type),
    param1 = .modify_priors_arg1(prior, type),
    param2 = .modify_priors_arg2(prior, type)
  )
}

.modify_priors_normalize_type <- function(type) {
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

.modify_priors_arg1 <- function(prior, type) {
  switch(
    .modify_priors_normalize_type(type),
    normal = .modify_priors_named_number(prior, c("mean", "mu", "param1")),
    uniform = .modify_priors_named_number(prior, c("min", "lower", "param1")),
    lognormal = .modify_priors_named_number(
      prior,
      c("meanlog", "mean", "param1")
    ),
    cauchy = .modify_priors_named_number(
      prior,
      c("location", "mean", "param1")
    ),
    beta = .modify_priors_named_number(prior, c("shape1", "alpha", "param1")),
    exponential = .modify_priors_named_number(
      prior,
      c("rate", "lambda", "param1")
    ),
    none = NaN
  )
}

.modify_priors_arg2 <- function(prior, type) {
  switch(
    .modify_priors_normalize_type(type),
    normal = .modify_priors_named_number(prior, c("sd", "sigma", "param2")),
    uniform = .modify_priors_named_number(prior, c("max", "upper", "param2")),
    lognormal = .modify_priors_named_number(
      prior,
      c("sdlog", "sd", "param2")
    ),
    cauchy = .modify_priors_named_number(prior, c("scale", "sd", "param2")),
    beta = .modify_priors_named_number(prior, c("shape2", "beta", "param2")),
    exponential = NaN,
    none = NaN
  )
}

.modify_priors_named_number <- function(x, keys) {
  for (key in keys) {
    if (!base::is.null(x[[key]])) {
      return(base::as.numeric(x[[key]][[1L]]))
    }
  }

  base::stop("Structured prior list is missing a required argument.")
}
