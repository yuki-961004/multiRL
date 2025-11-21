#' Upper-Confidence-Bound
#'
#' @param count count
#' @param params params
#' @param ... extra
#'
#' @returns bias
#' 
func_delta <- function(
    count,
    params,
    ...
){
  # if you need extra information
  # e.g.
  # Trial <- idinfo["Trial"]
  # Frame <- exinfo["Frame"]
  
  delta     <-  get_param(params, "delta")

  bias <- delta * sqrt(log(count + exp(1)) / (count + 1e-10))
  
  return(bias)
}
