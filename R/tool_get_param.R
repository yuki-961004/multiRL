.get_param <- function(x, name) {
  
  found <- c(
    purrr::pluck(x, "free", name),
    purrr::pluck(x, "fixed", name),
    purrr::pluck(x, "constant", name)
  )
  
  return(found)
}
