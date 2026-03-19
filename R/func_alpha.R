#' @title Function: Learning Rate
#' @description
#' 
#'  \deqn{Q_{new} = Q_{old} + \alpha \cdot (R - Q_{old})}
#'
#' @param shown
#'  Which options shown in this trial.
#' @param qvalue 
#'  The expected Q values of different behaviors produced by different systems 
#'    when updated to this trial.
#' @param reward 
#'  The feedback received by the agent from the environment at trial(t) 
#'    following the execution of action(a)
#' @param utility 
#'  The subjective value (internal representation) assigned by 
#'   the agent to the objective reward.
#' @param params 
#'  Parameters used by the model's internal functions,
#'    see \link[multiRL]{params}
#' @param system
#'  When the agent makes a decision, is a single system at work, or are multiple 
#'  systems involved?
#'    see \link[multiRL]{system} 
#' @param ... 
#'  It currently contains the following information; additional information 
#'    may be added in future package versions.
#' \itemize{
#'   \item idinfo: 
#'      \itemize{
#'        \item subid
#'        \item block
#'        \item trial
#'      }
#'   \item exinfo: 
#'      contains information whose column names are specified by the user.
#'      \itemize{
#'        \item Frame
#'        \item RT
#'        \item NetWorth
#'        \item ...
#'      }
#'   \item behave: 
#'      includes the following:
#'      \itemize{
#'        \item action: 
#'          the behavior performed by the human in the given trial.
#'        \item latent: 
#'          the object updated by the agent in the given trial.
#'        \item simulation: 
#'          the actual behavior performed by the agent.
#'        \item position:
#'          the position of the stimulus on the screen.
#'      }
#'    \item cue and rsp:
#'      Cues and responses within latent learning rules, 
#'        see \link[multiRL]{behrule} 
#'    \item state:
#'      The state stores the stimuli shown in the current trial—split into 
#'      components by underscores—and the rewards associated with them.
#' }
#'    
#' @return A \code{NumericVector} containing the updated values computed based 
#'    on the learning rate. 
#'    
#' @section Body: 
#' \preformatted{func_alpha <- function(
#'     shown,
#'     qvalue,
#'     reward,
#'     utility,
#'     params,
#'     system,
#'     ...
#' ){
#' 
#'   list2env(list(...), envir = environment())
#'   
#'   # If you need extra information(...)
#'   # Column names may be lost(C++), indexes are recommended
#'   # e.g.
#'   # Trial  <- idinfo[3]
#'   # Frame  <- exinfo[1]
#'   # Action <- behave[1]
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
    shown,
    qvalue,
    reward,
    utility,
    params,
    system,
    ...
){
  
  list2env(list(...), envir = environment())
  
  # If you need extra information(...)
  # Column names may be lost(C++), indexes are recommended
  # e.g.
  # Trial  <- idinfo[3]
  # Frame  <- exinfo[1]
  # Action <- behave[1]
  
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
