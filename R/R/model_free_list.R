.model_free_builtin <- function(
    name,
    free,
    params,
    priors,
    lower,
    upper,
    settings
) {
  if (base::is.null(priors)) {
    priors <- .model_free_default_priors(base::names(free))
  }
  if (base::is.null(lower)) {
    lower <- .model_free_default_lower(base::names(free))
  }
  if (base::is.null(upper)) {
    upper <- .model_free_default_upper(base::names(free))
  }
  if (base::is.null(settings)) {
    settings <- list()
  }
  if (base::is.null(params)) {
    params <- list(free = free)
  }

  list(
    model = name,
    process = "process_model_free",
    params = params,
    priors = priors,
    lower = lower,
    upper = upper,
    settings = utils::modifyList(
      x = list(name = name),
      val = settings
    )
  )
}

.model_free_default_priors <- function(names) {
  out <- list()
  for (name in names) {
    out[[name]] <- switch(
      name,
      beta = list(type = "exponential", rate = 1),
      list(type = "beta", shape1 = 2, shape2 = 2)
    )
  }
  out
}

.model_free_default_lower <- function(names) {
  out <- stats::setNames(base::rep(0, base::length(names)), names)
  base::as.list(out)
}

.model_free_default_upper <- function(names) {
  values <- base::vapply(
    names,
    function(name) {
      if (name == "beta") {
        return(5)
      }
      1
    },
    numeric(1L)
  )
  base::as.list(values)
}
