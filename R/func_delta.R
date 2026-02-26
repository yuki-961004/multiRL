#' @title Function: Upper-Confidence-Bound
#' @description
#' 
#'  \deqn{
#'    \text{Bias} = \delta \cdot \sqrt{\frac{\log(N + e)}{N + 10^{-10}}}
#'  }
#'  
#' @param count 
#'  How many times this action has been executed
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
#' @return A \code{NumericVector} containing the bias for each option based on 
#'    the number of times it has been selected.
#'    
#' @section Body: 
#' \preformatted{func_delta <- function(
#'     count,
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
#'   latent <- behave[2]
#'   position <- as.numeric(cue %in% latent)
#'   
#'   delta     <- params[["delta"]]
#'   sticky    <- params[["sticky"]]
#'   
#'   # Upper-Confidence-Bound
#'   bias <- delta * sqrt(log(count + exp(1)) / (count + 1e-10)) +
#'    # Sticky to the same response
#'    sticky * position
#'    
#'   return(bias)
#' }
#' }
#' 
func_delta <- function(
    count,
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
  latent <- behave[2]
  position <- as.numeric(cue %in% latent)
  
  delta     <- params[["delta"]]
  sticky    <- params[["sticky"]]
  
  # Upper-Confidence-Bound
  bias <- delta * sqrt(log(count + exp(1)) / (count + 1e-10)) + 
    # Sticky to the same response
    sticky * position
  
  return(bias)
}
