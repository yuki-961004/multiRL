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
#'  The estimated expected value of taking action(a) at trial(t) with bias.
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
#'  Subject ID, Block ID, Trial ID, and any additional information defined by 
#'    the user.
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
#'   # if you need extra information
#'   # e.g.
#'   # Trial <- idinfo["Trial"]
#'   # Frame <- exinfo["Frame"]
#'   
#'   beta      <-  multiRL:::get_param(params, "beta")
#'   lapse     <-  multiRL:::get_param(params, "lapse")
#'   weight    <-  get_param(params, "weight")
#'   
#'   n_system <- length(qvalue)
#'   
#'   if (length(weight) == 1L) {weight <- c(weight, 1 - weight)}
#'   
#'   weight <- weight / base::sum(weight)
#'   
#'   n_options <- length(qvalue[[1]])
#'   index     <- which(!is.na(qvalue[[1]]))
#'   n_shown   <- length(index)
#'   
#'   prob_list <- vector("list", length(qvalue))

#'   for (i in seq_along(qvalue)) {
#'     
#'     sub_qvalue <- qvalue[[i]]
#'     prob <- rep(NA_real_, n_options)
#'     
#'     if (explor == 1) {
#'       prob[index] <- 1 / n_shown
#'     } else {
#'       exp_stable <- exp(beta * (sub_qvalue - max(sub_qvalue, na.rm = TRUE)))
#'       prob <- exp_stable / sum(exp_stable, na.rm = TRUE)
#'     }
#'     
#'     prob_list[[i]] <- prob
#'   }
#'   
#'   prob <- Reduce(f = `+`, x = Map(`*`, prob_list, weight))
#'   prob <- as.vector(prob)
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
  # if you need extra information
  # e.g.
  # Trial <- idinfo["Trial"]
  # Frame <- exinfo["Frame"]
  
  beta      <-  get_param(params, "beta")
  lapse     <-  get_param(params, "lapse")
  weight    <-  get_param(params, "weight")
  
  n_system <- length(qvalue)
  
  if (length(weight) == 1L) {weight <- c(weight, 1 - weight)}
  
  weight <- weight / base::sum(weight)
  
  n_options <- length(qvalue[[1]])
  index     <- which(!is.na(qvalue[[1]]))
  n_shown   <- length(index)
  
  prob_list <- vector("list", length(qvalue))
  
  for (i in seq_along(qvalue)) {
    
    sub_qvalue <- qvalue[[i]]
    prob <- rep(NA_real_, n_options)
    
    if (explor == 1) {
      prob[index] <- 1 / n_shown
    } else {
      exp_stable <- exp(beta * (sub_qvalue - max(sub_qvalue, na.rm = TRUE)))
      prob <- exp_stable / sum(exp_stable, na.rm = TRUE)
    }
    
    prob_list[[i]] <- prob
  }
  
  prob <- Reduce(f = `+`, x = Map(`*`, prob_list, weight))
  prob <- as.vector(prob)
  
  # lapse
  prob <- (1 - lapse * n_shown) * prob + lapse
  
  return(prob)
}
