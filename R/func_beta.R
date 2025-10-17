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
  beta      <-  .get_param(params, "beta")
  
  n_options <- length(qvalue)
  prob      <- rep(x = NA_real_, times = n_options)
  index     <- which(!is.na(qvalue))
  n_shown   <- length(index)
  lapse     <- 0.01 * n_shown
  
  if (explor == 1) {
    # Exploration
    prob[index] <- 1 / n_shown
  } else {
    # Exploitation
    exp_stable <- exp(beta * (qvalue - max(qvalue, na.rm = TRUE)))
    prob <- exp_stable / sum(exp_stable, na.rm = TRUE)
  }
  
  # lapse
  prob <- (1 - lapse) * prob + (lapse / n_shown)
  
  return(prob)
}
