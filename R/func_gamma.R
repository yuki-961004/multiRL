#' @title Function: Utility Function
#' @description
#' 
#'  \deqn{U(R) = {R}^{\gamma}}
#'
#' @param shown
#'  Which options shown in this trial.
#' @param reward 
#'  The feedback received by the agent from the environment at trial(t) 
#'    following the execution of action(a)
#' @param params 
#'  Parameters used by the model's internal functions,
#'    see \link[multiRL]{params}
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
#'      }
#'    \item cue and rsp:
#'      Cues and responses within latent learning rules, 
#'        see \link[multiRL]{behrule} 
#' }
#'    
#' @return A \code{NumericVector} of length one representing the subjective 
#'    value transformed from the objective reward via the utility function.
#'    
#' @section Body: 
#' \preformatted{func_gamma <- function(
#'     shown,
#'     reward,
#'     params,
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
#'   gamma     <-  params[["gamma"]]
#'   
#'   # Stevens' Power Law
#'   utility <- ((reward >= 0) * 2 - 1) * (abs(reward) ^ gamma)
#'   
#'   return(utility)
#' }
#' }
#' 
func_gamma <- function(
    shown,
    reward,
    params,
    ...
){
  
  list2env(list(...), envir = environment())
  
  # If you need extra information(...)
  # Column names may be lost(C++), indexes are recommended
  # e.g.
  # Trial  <- idinfo[3]
  # Frame  <- exinfo[1]
  # Action <- behave[1]

  gamma     <-  params[["gamma"]]

  # Stevens' Power Law
  utility <- ((reward >= 0) * 2 - 1) * (abs(reward) ^ gamma)
  
  return(utility)
}
