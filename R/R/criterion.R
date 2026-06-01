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
  if (base::is.function(prior)) {
    return(.shell_prior_function_to_spec(prior))
  }

  if (base::is.list(prior) && !base::is.null(prior$type)) {
    return(.shell_prior_list_to_spec(prior))
  }

  base::stop(
    "Priors must be density functions or structured prior lists."
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

.shell_prior_function_to_spec <- function(prior) {
  call <- .shell_find_density_call(base::body(prior))
  if (base::is.null(call)) {
    base::stop(
      "Prior functions must call a supported stats density function."
    )
  }

  density_name <- .shell_density_name(call)
  type <- .shell_density_to_prior_type(density_name)
  args <- base::as.list(call)[-1L]
  env <- base::environment(prior)

  list(
    type = type,
    param1 = .shell_density_arg1(args, env, type),
    param2 = .shell_density_arg2(args, env, type)
  )
}

.shell_find_density_call <- function(expr) {
  if (!base::is.call(expr)) {
    return(NULL)
  }

  name <- .shell_density_name(expr)
  if (name %in% c("dnorm", "dunif", "dlnorm", "dcauchy", "dbeta", "dexp")) {
    return(expr)
  }

  children <- base::as.list(expr)[-1L]
  for (child in children) {
    found <- .shell_find_density_call(child)
    if (!base::is.null(found)) {
      return(found)
    }
  }

  NULL
}

.shell_density_name <- function(call) {
  if (!base::is.call(call)) {
    return("")
  }

  name <- base::as.character(call[[1L]])
  name[[base::length(name)]]
}

.shell_density_to_prior_type <- function(density_name) {
  switch(
    density_name,
    dnorm = "normal",
    dunif = "uniform",
    dlnorm = "lognormal",
    dcauchy = "cauchy",
    dbeta = "beta",
    dexp = "exponential",
    base::stop("Unsupported prior density function.")
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

.shell_density_arg1 <- function(args, env, type) {
  switch(
    type,
    normal = .shell_eval_density_arg(args, env, c("mean"), 2L, 0),
    uniform = .shell_eval_density_arg(args, env, c("min"), 2L, 0),
    lognormal = .shell_eval_density_arg(args, env, c("meanlog"), 2L, 0),
    cauchy = .shell_eval_density_arg(args, env, c("location"), 2L, 0),
    beta = .shell_eval_density_arg(args, env, c("shape1"), 2L),
    exponential = .shell_eval_density_arg(args, env, c("rate"), 2L, 1),
    base::stop("Unsupported prior type.")
  )
}

.shell_density_arg2 <- function(args, env, type) {
  switch(
    type,
    normal = .shell_eval_density_arg(args, env, c("sd"), 3L, 1),
    uniform = .shell_eval_density_arg(args, env, c("max"), 3L, 1),
    lognormal = .shell_eval_density_arg(args, env, c("sdlog"), 3L, 1),
    cauchy = .shell_eval_density_arg(args, env, c("scale"), 3L, 1),
    beta = .shell_eval_density_arg(args, env, c("shape2"), 3L),
    exponential = NaN,
    base::stop("Unsupported prior type.")
  )
}

.shell_eval_density_arg <- function(args, env, keys, position, default) {
  arg_names <- base::names(args)
  if (base::is.null(arg_names)) {
    arg_names <- base::rep("", base::length(args))
  }

  named_index <- base::match(keys, arg_names, nomatch = 0L)
  named_index <- named_index[named_index > 0L]
  if (base::length(named_index) > 0L) {
    return(base::as.numeric(base::eval(args[[named_index[[1L]]]], env)))
  }

  if (base::length(args) >= position) {
    return(base::as.numeric(base::eval(args[[position]], env)))
  }

  if (!base::missing(default)) {
    return(default)
  }

  base::stop("Prior density function is missing a required argument.")
}

.shell_named_number <- function(x, keys) {
  for (key in keys) {
    if (!base::is.null(x[[key]])) {
      return(base::as.numeric(x[[key]][[1L]]))
    }
  }

  base::stop("Structured prior list is missing a required argument.")
}
