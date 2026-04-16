#' @title Function: Bias
#' @description
#' 
#'  \deqn{
#'    \text{Bias} = \delta \cdot \sqrt{\frac{\log(N + e)}{N + 10^{-10}}}
#'  }
#'
#' @param shown
#'  Which options shown in this trial.
#' @param count 
#'  How many times this action has been executed
#' @param rownum 
#'  The trial number
#' @param params 
#'  Parameters used by the model's internal functions,
#'    see \link[multiRL]{params}
#' @param others
#'  All latent variables within the MDP process belong here.
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
#' @return A \code{NumericVector} containing the bias for each option based on 
#'    the number of times it has been selected.
#'    
#' @section Body: 
#' \preformatted{func_delta <- function(
#'     shown,
#'     count,
#'     rownum,
#'     params,
#'     others,
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
#'   # Sticky to the same latent
#'   latent <- behave[2]
#'   if (is.na(latent)) {
#'     last_latent <- shown * 0
#'   } else {
#'     last_latent <- as.numeric(!is.na(shown)) * as.numeric(cue \%in\% latent)
#'   }
#'   # Sticky to the same action(simulation)
#'   simulation <- behave[3]
#'   if (is.na(simulation)) {
#'     last_simulation <- shown * 0
#'   } else {
#'     last_simulation <- as.numeric(
#'       rowSums(state[shown, , drop = FALSE] == simulation) > 0
#'     )
#'   }
#'   # Sticky to the same position
#'   position <- behave[4]
#'   if (is.na(position)) {
#'     last_position <- shown * 0
#'   } else {
#'     last_position <- as.numeric(shown == as.numeric(position))
#'   }
#'   
#'   delta     <- params[["delta"]]
#'   sticky    <- params[["sticky"]]
#'   
#'   # Upper-Confidence-Bound
#'   bias <- delta * sqrt(log(count + exp(1)) / (count + 1e-10)) + 
#'     # Sticky to the same latent
#'     sticky * last_latent +
#'     # Sticky to the same action(simulation)
#'     sticky * last_simulation +
#'     # Sticky to the same position
#'     sticky * last_position 
#'   
#'   return(list(bias = bias, others = others)) 
#' }
#' }
#' 
func_delta <- function(
    shown,
    count,
    rownum,
    params,
    others,
    ...
){
  
  list2env(list(...), envir = environment())
  
  # If you need extra information(...)
  # Column names may be lost(C++), indexes are recommended
  # e.g.
  # Trial  <- idinfo[3]
  # Frame  <- exinfo[1]
  # Action <- behave[1]
  
  # Sticky to the same latent
  latent <- behave[2]
  if (is.na(latent)) {
    last_latent <- shown * 0
  } else {
    last_latent <- as.numeric(!is.na(shown)) * as.numeric(cue %in% latent)
  }
  # Sticky to the same action(simulation)
  simulation <- behave[3]
  if (is.na(simulation)) {
    last_simulation <- shown * 0
  } else {
    last_simulation <- as.numeric(
      rowSums(state[shown, , drop = FALSE] == simulation) > 0
    )
  }
  # Sticky to the same position
  position <- behave[4]
  if (is.na(position)) {
    last_position <- shown * 0
  } else {
    last_position <- as.numeric(shown == as.numeric(position))
  }
  
  delta     <- params[["delta"]]
  sticky    <- params[["sticky"]]
  
  # Upper-Confidence-Bound
  bias <- delta * sqrt(log(count + exp(1)) / (count + 1e-10)) + 
    # Sticky to the same latent
    sticky * last_latent +
    # Sticky to the same action(simulation)
    sticky * last_simulation +
    # Sticky to the same position
    sticky * last_position

  return(list(bias = bias, others = others)) 
}
