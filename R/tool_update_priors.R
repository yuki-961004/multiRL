.update_priors <- function(x, priors) {
  
  free_params <- list()
  posteriors <- list()
  
  # 把自由参数找出来
  for (i in 1:length(x)) {
    free_params[[i]] <- x[[i]]@input@params@free
  }
  # 把自由参数变成表格
  free_params <- dplyr::bind_rows(free_params)
  # 按照求得的自由参数最佳值, 按照正态分布, 计算后验分布
  for (i in 1:length(priors)) {
    mean <- base::mean(free_params[[i]])
    sd <- stats::sd(free_params[[i]]) + 1e-5
    
    # 创建只包含 mean 和 sd 的干净环境
    fn_env <- base::new.env(parent = base::baseenv())
    fn_env$mean <- mean
    fn_env$sd <- sd
    
    # 定义函数
    posteriors[[i]] <- function(x) {
      stats::dnorm(x, log = TRUE, mean = mean, sd = sd)
    }
    base::environment(posteriors[[i]]) <- fn_env
  }
  names(posteriors) <- names(priors)
  
  
  return(posteriors)
}
