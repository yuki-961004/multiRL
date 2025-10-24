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
  alpha_n   <-  .get_param(params, "alpha_n")
  alpha_p   <-  .get_param(params, "alpha_p")
  
  if (
    !(is.null(alpha)) && is.null(alpha_n) && is.null(alpha_p)
  ) {model <- "TD"} 
  else if (
    is.null(alpha) && !(is.null(alpha_n)) && !(is.null(alpha_p))
  ) {model <- "RSTD"} 
  else {stop("Unknown Model! Plase modify your learning rate function")}

  # TD
  if (model == "TD") {
    update <- qvalue + alpha   * (reward - qvalue)
  # RSTD
  } 
  else if (model == "RSTD" && reward < qvalue) {
    update <- qvalue + alpha_n * (reward - qvalue)
  } 
  else if (model == "RSTD" && reward >= qvalue) {
    update <- qvalue + alpha_p * (reward - qvalue)
  }

  return(update)
}
