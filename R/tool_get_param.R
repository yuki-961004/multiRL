.get_param <- function(x, name) {
  
  found <- c(
    purrr::pluck(x, "free", name),
    purrr::pluck(x, "fixed", name),
    purrr::pluck(x, "constant", name)
  )
  
  if (length(found) == 0) {
    stop(sprintf("Parameter '%s' not found in any source.", name))
  } 
  
  return(found)
}
