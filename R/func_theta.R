#' decay function
#'
#' @param value0 values
#' @param values values
#' @param reward reward
#' @param params params
#' @param ... extra
#'
#' @returns decayed values
#' 
func_theta <- function(
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
  
  theta      <-  get_param(params, "theta")
  bonus      <-  get_param(params, "bonus")
  
  if (reward == 0) {
    decay <- values + theta * (value0 - values)
  } else if (reward < 0) {
    decay <- values + theta * (value0 - values) + bonus
  } else if (reward > 0) {
    decay <- values + theta * (value0 - values) - bonus
  }

  return(decay)
}
