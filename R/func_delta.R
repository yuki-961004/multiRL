#' @title Function: Upper-Confidence-Bound
#' @description
#' 
#'  \deqn{
#'    \text{Bias} = \delta \cdot \sqrt{\frac{\log(N + e)}{N + 10^{-10}}}
#'  }
#'  
#' @param count 
#'  How many times this action has been executed
#' @param params 
#'  Parameters used by the model’s internal functions,
#'    see \link[multiRL]{params}
#' @param ... 
#'  Subject ID, Block ID, Trial ID, and any additional information defined by 
#'    the user.
#'    
#' @section Body: 
#' \preformatted{func_delta <- function(
#'     count,
#'     params,
#'     ...
#' ){
#'   # if you need extra information
#'   # e.g.
#'   # Trial <- idinfo["Trial"]
#'   # Frame <- exinfo["Frame"]
#'   
#'   delta     <-  get_param(params, "delta")
#'   
#'   bias <- delta * sqrt(log(count + exp(1)) / (count + 1e-10))
#'   
#'   return(bias)
#' }
#' }
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
