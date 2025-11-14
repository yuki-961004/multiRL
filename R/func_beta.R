#' soft max
#'
#' @param qvalue qvalue
#' @param explor explor
#' @param params params
#' @param ... extra
#'
#' @returns probability
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
  
  beta      <-  .get_param(params, "beta")
  lapse     <-  .get_param(params, "lapse")
  
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
