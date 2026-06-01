.shell_modify_params <- function(x, val) {
  params <- utils::modifyList(x = x, val = val)

  free_names <- base::names(params$free)

  keep_in_fixed <- base::setdiff(
    x = base::names(params$fixed),
    y = free_names
  )
  params$fixed <- params$fixed[keep_in_fixed]

  keep_in_constant <- base::setdiff(
    x = base::names(params$constant),
    y = free_names
  )
  params$constant <- params$constant[keep_in_constant]

  params
}

.shell_flatten_params <- function(params) {
  flat <- base::c(params$free, params$fixed, params$constant)
  base::unlist(flat, use.names = TRUE)
}
