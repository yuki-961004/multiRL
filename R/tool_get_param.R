get_param <- function(
    x, name
) {
  if (!is(x, "multiRL.params")) {
    stop("'x' must be an object of class 'multiRL.params'.")
  }
  
  param <- purrr::pluck(x, "free", name) %||% purrr::pluck(x, "fixed", name)
  
  return(param)
}
