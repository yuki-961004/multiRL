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
  alphaN    <-  .get_param(params, "alphaN")
  alphaP    <-  .get_param(params, "alphaP")
  
  if (
    !(is.null(alpha)) && is.null(alphaN) && is.null(alphaP)
  ) {model <- "TD"} 
  else if (
    is.null(alpha) && !(is.null(alphaN)) && !(is.null(alphaP))
  ) {model <- "RSTD"} 
  else {stop("Unknown Model! Plase modify your learning rate function")}

  # TD
  if (model == "TD") {
    update <- qvalue + alpha   * (reward - qvalue)
  # RSTD
  } 
  else if (model == "RSTD" && reward < qvalue) {
    update <- qvalue + alphaN * (reward - qvalue)
  } 
  else if (model == "RSTD" && reward >= qvalue) {
    update <- qvalue + alphaP * (reward - qvalue)
  }

  return(update)
}
