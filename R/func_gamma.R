#' @title Function: Utility Function
#' @description
#' 
#'  \deqn{U(R) = {R}^{\gamma}}
#'
#' @param reward 
#'  The feedback received by the agent from the environment at trial(t) 
#'    following the execution of action(a)
#' @param params 
#'  Parameters used by the model’s internal functions,
#'    see \link[multiRL]{params}
#' @param ... 
#'  Subject ID, Block ID, Trial ID, and any additional information defined by 
#'    the user.
#'    
#' @section Body: 
#' \preformatted{func_gamma <- function(
#'     reward,
#'     params,
#'     ...
#' ){
#'   # if you need extra information
#'   # e.g.
#'   # Trial <- idinfo["Trial"]
#'   # Frame <- exinfo["Frame"]
#'   
#'   gamma     <-  multiRL:::get_param(params, "gamma")
#'   
#'   # Stevens' Power Law
#'   utility <- sign(reward) * (abs(reward) ^ gamma)
#'   
#'   return(utility)
#' }
#' }
#' 
func_gamma <- function(
    reward,
    params,
    ...
){
  # if you need extra information
  # e.g.
  # Trial <- idinfo["Trial"]
  # Frame <- exinfo["Frame"]

  gamma     <-  get_param(params, "gamma")

  # Stevens' Power Law
  utility <- sign(reward) * (abs(reward) ^ gamma)
  
  return(utility)
}
