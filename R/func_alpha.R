#' @title Function: Learning Rate
#' @description
#' 
#'  \deqn{Q_{new} = Q_{old} + \alpha \cdot (R - Q_{old})}
#'
#' @param qvalue 
#'  The estimated expected value of taking action(a) at trial(t).
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
#' \preformatted{func_alpha <- function(
#'     qvalue,
#'     reward,
#'     params,
#'     ...
#' ){
#'   # if you need extra information
#'   # e.g.
#'   # Trial <- idinfo["Trial"]
#'   # Frame <- exinfo["Frame"]
#'   
#'   alpha     <-  multiRL:::get_param(params, "alpha")
#'   alphaN    <-  multiRL:::get_param(params, "alphaN")
#'   alphaP    <-  multiRL:::get_param(params, "alphaP")
#'   
#'   # Determine the model currently in use based on which parameters are free.
#'   if (
#'     !(is.na(alpha)) && is.na(alphaN) && is.na(alphaP)
#'   ) {
#'     model <- "TD"
#'   } else if (
#'     is.na(alpha) && !(is.na(alphaN)) && !(is.na(alphaP))
#'   ) {
#'     model <- "RSTD"
#'   } else {
#'     stop("Unknown Model! Plase modify your learning rate function")
#'   }
#'   
#'   # TD
#'   if (model == "TD") {
#'     update <- qvalue + alpha   * (reward - qvalue)
#'     # RSTD
#'   } else if (model == "RSTD" && reward < qvalue) {
#'     update <- qvalue + alphaN * (reward - qvalue)
#'   } else if (model == "RSTD" && reward >= qvalue) {
#'     update <- qvalue + alphaP * (reward - qvalue)
#'   }
#'   
#'   return(update)
#' }
#' }
func_alpha <- function(
    qvalue,
    reward,
    params,
    ...
){
  # if you need extra information
  # e.g.
  # Trial <- idinfo["Trial"]
  # Frame <- exinfo["Frame"]
  
  alpha     <-  get_param(params, "alpha")
  alphaN    <-  get_param(params, "alphaN")
  alphaP    <-  get_param(params, "alphaP")
  
  # Determine the model currently in use based on which parameters are free.
  if (
    !(is.na(alpha)) && is.na(alphaN) && is.na(alphaP)
  ) {
    model <- "TD"
  } else if (
    is.na(alpha) && !(is.na(alphaN)) && !(is.na(alphaP))
  ) {
    model <- "RSTD"
  } else {
    stop("Unknown Model! Plase modify your learning rate function")
  }

  # TD
  if (model == "TD") {
    update <- qvalue + alpha   * (reward - qvalue)
  # RSTD
  } else if (model == "RSTD" && reward < qvalue) {
    update <- qvalue + alphaN * (reward - qvalue)
  } else if (model == "RSTD" && reward >= qvalue) {
    update <- qvalue + alphaP * (reward - qvalue)
  }

  return(update)
}
