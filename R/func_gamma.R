#' Utility Function
#'
#' @param reward reward
#' @param params params
#' @param ... extra
#'
#' @returns utility
#' 
func_gamma <- function(
    reward,
    params,
    ...
){
  gamma     <-  .get_param(params, "gamma")

  utility <- sign(reward) * (abs(reward) ^ gamma)
}
