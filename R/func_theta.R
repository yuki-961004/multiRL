#' decay function
#'
#' @param value1 values
#' @param values values
#' @param params params
#' @param ... extra
#'
#' @returns decayed values
#' 
func_theta <- function(
    value1, 
    values,
    params,
    ...
){
  # if you need extra information
  # e.g.
  # Trial <- idinfo["Trial"]
  # Frame <- exinfo["Frame"]
  
  theta      <-  get_param(params, "theta")
  
  decay      <- values + theta * (value1 - values)
  
  return(decay)
}
