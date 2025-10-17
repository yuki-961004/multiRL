.get_param <- function(x, name) {
  
  param <- purrr::pluck(x, "free", name) %||% purrr::pluck(x, "fixed", name)
  
  return(param)
}
