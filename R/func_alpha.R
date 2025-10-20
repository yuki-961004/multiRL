#' learning rate
#'
#' @param qvalue qvalue
#' @param reward reward
#' @param params params
#' @param ... extra
#'
#' @returns update
#' 
func_alpha <- function(
    qvalue,
    reward,
    params,
    ...
){
  alpha     <-  .get_param(params, "alpha")
  
  # TD
  if (length(alpha) == 1) {
    update <- qvalue + alpha * (reward - qvalue)
  # RSTD
  } 
  else if (length(alpha) == 2 & reward < qvalue) {
    update <- qvalue + alpha[1] * (reward - qvalue)
  } 
  else if (length(alpha) == 2 & reward >= qvalue) {
    update <- qvalue + alpha[2] * (reward - qvalue)
  }

  return(update)
}
