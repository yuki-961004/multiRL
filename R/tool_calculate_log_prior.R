.calculate_log_prior = function(priors, params) {
  
  log_prior <- mapply(
    FUN = function(params, priors) {sapply(X = params, FUN = priors)},
    params = params,
    priors = priors,
    SIMPLIFY = FALSE 
  )

  sum_log_prior <- sum(unlist(log_prior))
  
  return(sum_log_prior)
}
