#' @title Function: Decay Rate
#' @description
#' 
#'  \deqn{W_{new} = W_{old} + \zeta \cdot (W_{0} - W_{old})}
#'
#' @param value0 
#'  The initial values for all actions.
#' @param values 
#'  The current expected values for all actions.
#' @param reward 
#'  The feedback received by the agent from the environment at trial(t) 
#'    following the execution of action(a)
#' @param params 
#'  Parameters used by the model’s internal functions,
#'    see \link[multiRL]{params}
#' @param system
#'  When the agent makes a decision, is a single system at work, or are multiple 
#'  systems involved?
#'    see \link[multiRL]{system} 
#' @param ... 
#'  Subject ID, Block ID, Trial ID, and any additional information defined by 
#'    the user.
#'    
#' @return A \code{NumericVector} representing the values of unchosen options 
#'    after decay according to the decay rate.
#'    
#' @section Body: 
#' \preformatted{func_zeta <- function(
#'     value0, 
#'     values,
#'     reward,
#'     params,
#'     ...
#' ){
#'   # if you need extra information
#'   # e.g.
#'   # Trial <- idinfo["Trial"]
#'   # Frame <- exinfo["Frame"]
#'   
#'   zeta       <-  multiRL:::get_param(params, "zeta")
#'   bonus      <-  multiRL:::get_param(params, "bonus")
#'   
#'   if (reward == 0) {
#'     decay <- values + zeta * (value0 - values)
#'   } else if (reward < 0) {
#'     decay <- values + zeta * (value0 - values) + bonus
#'   } else if (reward > 0) {
#'     decay <- values + zeta * (value0 - values) - bonus
#'   }
#'   
#'   return(decay)
#' }
#' }
#' 
func_zeta <- function(
    value0, 
    values,
    reward,
    params,
    system,
    ...
){
  # if you need extra information
  # e.g.
  # Trial <- idinfo["Trial"]
  # Frame <- exinfo["Frame"]
  
  zeta       <-  get_param(params, "zeta")
  bonus      <-  get_param(params, "bonus")
  
  if (reward == 0) {
    decay <- values + zeta * (value0 - values)
  } else if (reward < 0) {
    decay <- values + zeta * (value0 - values) + bonus
  } else if (reward > 0) {
    decay <- values + zeta * (value0 - values) - bonus
  }

  return(decay)
}
