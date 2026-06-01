.modify_params <- function(params) {
  default_params <- list(
    free = list(),
    fixed = list(
      gamma = 1,
      delta = 0.1,
      epsilon = NA_real_,
      zeta = 0
    ),
    constant = list(
      seed = 123,
      chunk = NULL,
      L = NA_real_,
      penalty = 1,
      Q0 = NaN,
      reset = NaN,
      lapse = 0.01,
      threshold = 1,
      bonus = 0,
      weight = 1,
      capacity = 0,
      sticky = 0
    )
  )

  out <- utils::modifyList(x = default_params, val = params)
  free_names <- base::names(out$free)

  out$fixed <- out$fixed[
    base::setdiff(base::names(out$fixed), free_names)
  ]
  out$constant <- out$constant[
    base::setdiff(base::names(out$constant), free_names)
  ]
  out$flat <- .modify_params_flatten(out)
  out
}

.modify_params_flatten <- function(params) {
  flat <- base::c(params$free, params$fixed, params$constant)
  base::unlist(flat, use.names = TRUE)
}
