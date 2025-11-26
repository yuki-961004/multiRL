.calculate_log_prior = function(priors, params) {
  
  # 根据先验概率密度函数, 计算自由参数的最佳值对应的Lpo
  log_prior <- mapply(
    FUN = function(params, priors) {sapply(X = params, FUN = priors)},
    params = params,
    priors = priors,
    SIMPLIFY = FALSE 
  )

  sum_log_prior <- sum(unlist(log_prior))
  
  return(sum_log_prior)
}
