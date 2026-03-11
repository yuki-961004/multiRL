#' @title Function: Decay Rate
#' @description
#' 
#'  \deqn{W_{new} = W_{old} + \zeta \cdot (W_{0} - W_{old})}
#'
#' @param shown
#'  Which options shown in this trial.
#' @param value0 
#'  The initial values for all actions.
#' @param values 
#'  The current expected values for all actions.
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
#'      }
#'    \item cue and rsp:
#'      Cues and responses within latent learning rules, 
#'        see \link[multiRL]{behrule} 
#' }
#'    
#' @return A \code{NumericVector} representing the values of unchosen options 
#'    after decay according to the decay rate.
#'    
#' @section Body: 
#' \preformatted{func_zeta <- function(
#'     value0, 
#'     values,
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
#'   zeta       <-  params[["zeta"]]
#'   bonus      <-  params[["bonus"]]
#'   
#'   if (reward == 0) {
#'     decay <- values + zeta * (value0 - values)
#'   } else if (reward < 0) {
#'     decay <- values + zeta * (value0 - values) + bonus
#'   } else if (reward > 0) {
#'     decay <- values + zeta * (value0 - values) - bonus
#'   }
#'   
#'   return(decay)
#' }
#' }
#' 
func_zeta <- function(
    shown,
    value0, 
    values,
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
  
  zeta       <-  params[["zeta"]]
  bonus      <-  params[["bonus"]]
  
  if (reward == 0) {
    decay <- values + zeta * (value0 - values)
  } else if (reward < 0) {
    decay <- values + zeta * (value0 - values) + bonus
  } else if (reward > 0) {
    decay <- values + zeta * (value0 - values) - bonus
  }

  return(decay)
}
