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
#' @param system
#'  When the agent makes a decision, is a single system at work, or are multiple 
#'  systems involved?
#'    see \link[multiRL]{system} 
#' @param ... 
#'  Subject ID, Block ID, Trial ID, and any additional information defined by 
#'    the user.
#'    
#' @return A \code{NumericVector} containing the updated values computed based 
#'    on the learning rate. 
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
#'   alpha     <-  params[["alpha"]]
#'   alphaN    <-  params[["alphaN"]]
#'   alphaP    <-  params[["alphaP"]]
#'   
#'   # Determine the model currently in use based on which parameters are free.
#'   if (
#'     system == "RL" && !(is.null(alpha)) && is.null(alphaN) && is.null(alphaP)
#'   ) {
#'     model <- "TD"
#'   } else if (
#'     system == "RL" && is.null(alpha) && !(is.null(alphaN)) && !(is.null(alphaP))
#'   ) {
#'     model <- "RSTD"
#'   } else if (
#'     system == "WM"
#'   ) {
#'     model <- "WM"
#'     alpha <- 1
#'   } else {
#'     stop("Unknown Model! Plase modify your learning rate function")
#'   }
#'   
#'   # TD
#'   if (model == "TD") {
#'     update <- qvalue + alpha   * (reward - qvalue)
#'   # RSTD
#'   } else if (model == "RSTD" && reward < qvalue) {
#'     update <- qvalue + alphaN * (reward - qvalue)
#'   } else if (model == "RSTD" && reward >= qvalue) {
#'     update <- qvalue + alphaP * (reward - qvalue)
#'   # WM
#'   } else if (model == "WM") {
#'     update <- qvalue + alpha  * (reward - qvalue)
#'   }
#'   
#'   return(update)
#' }
#' }
#' 
func_alpha <- function(
    qvalue,
    reward,
    params,
    system,
    ...
){
  # if you need extra information
  # e.g.
  # Trial <- idinfo["Trial"]
  # Frame <- exinfo["Frame"]

  alpha     <-  params[["alpha"]]
  alphaN    <-  params[["alphaN"]]
  alphaP    <-  params[["alphaP"]]
  
  # Determine the model currently in use based on which parameters are free.
  if (
    system == "RL" && !(is.null(alpha)) && is.null(alphaN) && is.null(alphaP)
  ) {
    model <- "TD"
  } else if (
    system == "RL" && is.null(alpha) && !(is.null(alphaN)) && !(is.null(alphaP))
  ) {
    model <- "RSTD"
  } else if (
    system == "WM"
  ) {
    model <- "WM"
    alpha <- 1
  } else {
    stop("Unknown Model! Plase modify your learning rate function")
  }

  # TD
  if (model == "TD") {
    update <- qvalue + alpha  * (reward - qvalue)
  # RSTD
  } else if (model == "RSTD" && reward < qvalue) {
    update <- qvalue + alphaN * (reward - qvalue)
  } else if (model == "RSTD" && reward >= qvalue) {
    update <- qvalue + alphaP * (reward - qvalue)
  # WM
  } else if (model == "WM") {
    update <- qvalue + alpha  * (reward - qvalue)
  }

  return(update)
}
