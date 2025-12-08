#' Function: Decay Rate
#'
#' @param value0 values
#' @param values values
#' @param reward reward
#' @param params params
#' @param ... extra
#'
#' @returns decayed values
#' 
func_zeta <- function(
    value0, 
    values,
    reward,
    params,
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
