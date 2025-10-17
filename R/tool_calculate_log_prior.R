.calculate_log_prior = function(priors, params) {
  
  log_densities <- mapply(
    FUN = function(param_values, prior_fn) {
      sapply(
        X = param_values,
        FUN = prior_fn
      )
    },
    param_values = params,
    prior_fn = priors,
    SIMPLIFY = FALSE 
  )

  log_prior_sum <- sum(unlist(log_densities))
  
  return(log_prior_sum)
}
