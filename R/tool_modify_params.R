.modify_params <- function(x, val) {
  
  params <- utils::modifyList(x = x, val = val)

  free_names <- names(params$free)
  
  keep_in_fixed <- base::setdiff(x = names(params$fixed), y = free_names)
  params$fixed <- params$fixed[keep_in_fixed]
  
  keep_in_constant <- base::setdiff(x = names(params$constant), y = free_names)
  params$constant <- params$constant[keep_in_constant]
  
  return(params)
}
