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
#' @return A \code{NumericVector} containing the bias for each option based on 
#'    the number of times it has been selected.
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
#'   delta     <-  params[["delta"]]
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
  
  list2env(list(...), envir = environment())
  
  # if you need extra information
  # e.g.
  # Trial <- idinfo[3]
  # Frame <- exinfo[1]
  
  delta     <-  params[["delta"]]

  bias <- delta * sqrt(log(count + exp(1)) / (count + 1e-10))
  
  return(bias)
}
