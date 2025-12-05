.calculate_log_prior = function(priors, params) {
  
  # 初始化累加器
  sum_log_prior <- 0.0
  
  # 根据先验概率密度函数, 计算自由参数的最佳值对应的Lpo
  for (i in seq_along(params)) {
    sum_log_prior <- sum_log_prior + sum(priors[[i]](params[[i]]))
  }
  
  return(sum_log_prior)
}
