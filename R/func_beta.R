#' @title Function: Soft-Max
#' @description
#' 
#'  \deqn{
#'    P_{t}(a) = 
#'    \frac{
#'      \exp(\beta \cdot (Q_t(a) - \max_{a' \in \mathcal{A}} Q_t(a')))
#'    }
#'    {
#'      \sum_{a' \in \mathcal{A}} 
#'      \exp(
#'        \beta \cdot (Q_t(a') - \max_{a'_{i} \in \mathcal{A}} Q_t(a'_{i}))
#'      )
#'    }
#'  }
#'    
#'  \deqn{
#'    P_{t}(a) = (1 - lapse \cdot N_{shown}) \cdot P_{t}(a) + lapse
#'  }
#'
#' @param qvalue 
#'  The expected Q values of different behaviors produced by different systems 
#'    when updated to this trial.
#' @param explor 
#'  Whether the agent made a random choice (exploration) in this trial.
#' @param params 
#'  Parameters used by the model’s internal functions,
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
#' }
#'    
#' @return A \code{NumericVector} containing the probability of choosing each 
#'    option.
#'    
#' @section Body: 
#' \preformatted{func_beta <- function(
#'     qvalue, 
#'     explor,
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
#'   beta     <- params[["beta"]]
#'   lapse    <- params[["lapse"]]
#'   weight   <- params[["weight"]]
#'   capacity <- params[["capacity"]]
#'   sticky   <- params[["sticky"]]
#'   
#'   index     <- which(!is.na(qvalue[[1]]))
#'   n_shown   <- length(index)
#'   n_system  <- length(qvalue)
#'   n_options <- length(qvalue[[1]])
#'   
#'   # Assign weights to different systems
#'   if (length(weight) == 1L) {weight <- c(weight, 1 - weight)}
#'   weight <- weight / sum(weight)
#'   if (n_system == 1) {weight <- weight[1]}
#'   
#'   # Compute the probabilities estimated by different systems
#'   prob_mat <- matrix(0, nrow = n_options, ncol = n_system)
#'   
#'   if (explor == 1) {
#'     prob_mat[index, ] <- 1 / n_shown
#'   } else {
#'     for (s in seq_len(n_system)) {
#'       sub_qvalue <- qvalue[[s]]
#'       exp_stable <- exp(beta * (sub_qvalue - max(sub_qvalue, na.rm = TRUE)))
#'       prob_mat[, s] <- exp_stable / sum(exp_stable, na.rm = TRUE)
#'     }
#'   }
#'   
#'   # Weighted average
#'   prob <- as.vector(prob_mat %*% weight)
#'   
#'   # lapse
#'   prob <- (1 - lapse * n_shown) * prob + lapse
#'   
#'   return(prob)
#' }
#' }
#' 
func_beta <- function(
    qvalue, 
    explor,
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
  
  beta     <- params[["beta"]]
  lapse    <- params[["lapse"]]
  weight   <- params[["weight"]]
  capacity <- params[["capacity"]]
  sticky   <- params[["sticky"]]

  index     <- which(!is.na(qvalue[[1]]))
  n_shown   <- length(index)
  n_system  <- length(qvalue)
  n_options <- length(qvalue[[1]])
  
  # Assign weights to different systems
  if (length(weight) == 1L) {weight <- c(weight, 1 - weight)}
  weight <- weight / sum(weight)
  if (n_system == 1) {weight <- weight[1]}
  
  # Compute the probabilities estimated by different systems
  prob_mat <- matrix(0, nrow = n_options, ncol = n_system)
  
  if (explor == 1) {
    prob_mat[index, ] <- 1 / n_shown
  } else {
    for (s in seq_len(n_system)) {
      sub_qvalue <- qvalue[[s]]
      exp_stable <- exp(beta * (sub_qvalue - max(sub_qvalue, na.rm = TRUE)))
      prob_mat[, s] <- exp_stable / sum(exp_stable, na.rm = TRUE)
    }
  }
  
  # Weighted average
  prob <- as.vector(prob_mat %*% weight)
  
  # lapse
  prob <- (1 - lapse * n_shown) * prob + lapse
  
  return(prob)
}
