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
#' @param ... 
#'  Subject ID, Block ID, Trial ID, and any additional information defined by 
#'    the user.
#' 
func_beta <- function(
    qvalue, 
    explor,
    params,
    ...
){
  # if you need extra information
  # e.g.
  # Trial <- idinfo["Trial"]
  # Frame <- exinfo["Frame"]
  
  beta      <-  get_param(params, "beta")
  lapse     <-  get_param(params, "lapse")
  
  n_options <- length(qvalue)
  prob      <- rep(x = NA_real_, times = n_options)
  index     <- which(!is.na(qvalue))
  n_shown   <- length(index)
  
  if (explor == 1) {
    # Exploration
    prob[index] <- 1 / n_shown
  } else {
    # Exploitation
    exp_stable <- exp(beta * (qvalue - max(qvalue, na.rm = TRUE)))
    prob <- exp_stable / sum(exp_stable, na.rm = TRUE)
  }
  
  # lapse
  prob <- (1 - lapse * n_shown) * prob + lapse
  
  return(prob)
}
