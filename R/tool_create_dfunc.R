.create_dfunc <- function(df_params, func_name) {
  
  # 1. 初始化一个干净的环境用于封装参数（Closure）
  fn_env <- base::new.env(parent = base::baseenv())
  
  # 2. 根据 func_name 计算参数并设定函数体
  # 使用 base::switch 避免长串 if-else
  post_func <- base::switch(
    func_name,
    # --- 均匀分布 (unif) ---
    "unif" = {
      
      # 计算参数
      mean        <- base::mean(df_params)
      sd          <- stats::sd(df_params) + 1e-5
      # 传入环境
      fn_env$mean <- mean
      fn_env$sd   <- sd
      
      # 定义函数体
      function(x) {
        stats::dnorm(
          x, 
          log = TRUE, 
          mean = mean, 
          sd = sd
        )
      }
    },
    
    # --- 正态分布 (norm) ---
    "norm" = {
      
      # 计算参数
      mean        <- base::mean(df_params)
      sd          <- stats::sd(df_params) + 1e-5
      # 传入环境
      fn_env$mean <- mean
      fn_env$sd   <- sd
      
      # 定义函数体
      function(x) {
        stats::dnorm(
          x, 
          log = TRUE, 
          mean = mean, 
          sd = sd
        )
      }
    },
    
    # --- Beta 分布 (beta) ---
    "beta" = {
      
      # Beta 分布参数估计：使用矩匹配法 (Method of Moments)
      # ⚠️ 严格要求：Beta 分布数据必须在 (0, 1) 区间内
      if (base::min(df_params) <= 0 || base::max(df_params) >= 1) {
        base::stop("Data for Beta distribution must be in (0, 1).")
      }
      
      # 计算参数
      mean <- base::mean(df_params)
      var  <- stats::var(df_params)
      shape1 <- mean * (mean * (1 - mean) / var - 1)
      shape2 <- (1 - mean) * (mean * (1 - mean) / var - 1)
      
      # 防止极端值
      if (shape1 <= 0 || shape2 <= 0) {
        warning(
          "Beta parameter estimation failed. Using shape1=2, shape2=2."
        )
        shape1 <- 2
        shape2 <- 2
      }
      
      # 传入环境
      fn_env$shape1 <- shape1
      fn_env$shape2 <- shape2
      
      # 定义函数体
      function(x) {
        stats::dbeta(
          x, 
          log = TRUE, 
          shape1 = shape1, 
          shape2 = shape2
        )
      }
    },
    
    # --- 指数分布 (exp) ---
    "exp" = {
      
      # 指数分布参数估计：rate = 1 / mean
      # 严格要求：指数分布数据必须 > 0
      if (base::min(df_params) <= 0) {
        base::stop("Data for Exponential distribution must be > 0.")
      }
      
      # 计算参数
      rate <- 1 / base::mean(df_params)
      # 传入环境
      fn_env$rate <- 1 / base::mean(df_params)
      
      # 定义函数体
      function(x) {
        stats::dexp(
          x, 
          log = TRUE, 
          rate = rate
        )
      }
    },
    
    # --- Gamma 分布 (gamma) ---
    "gamma" = {
      
      # 严格要求：Gamma 分布数据必须 > 0
      if (base::min(df_params) <= 0) {
        base::stop("Data for Gamma distribution must be > 0.")
      }
      
      # 计算参数
      mean <- base::mean(df_params)
      var  <- stats::var(df_params)
      shape <- (mean ^ 2) / var
      rate  <- mean / var
      # 如果估计的形状参数不合理 (如 <= 0)，使用默认值
      if (shape <= 0) {
        warning(
          "Gamma parameter estimation failed. Using shape=1, rate=1 (Exponential)."
        )
        shape <- 1
        rate  <- 1
      }
      # 传入环境
      fn_env$shape <- shape
      fn_env$rate <- rate
      
      # 定义函数体
      function(x) {
        stats::dgamma(
          x,
          log = TRUE,
          shape = shape,
          rate = rate
        )
      }
    },
    
    # --- 对数正态分布 (lnorm) ---
    "lnorm" = {
      
      # 严格要求：Log-Normal 分布数据必须 > 0
      if (base::min(df_params) <= 0) {
        base::stop("Data for Log-Normal distribution must be > 0.")
      }
      # 计算参数
      log_data <- base::log(df_params)
      meanlog <- base::mean(log_data)
      sdlog   <- stats::sd(log_data) + 1e-5
      # 传入环境
      fn_env$meanlog <- meanlog
      fn_env$sdlog   <- sdlog
      # 定义函数体
      function(x) {
        stats::dlnorm(
          x,
          log = TRUE,
          meanlog = meanlog,
          sdlog = sdlog
        )
      }
    },
    
    # --- 默认情况 (处理未知的 func_name) ---
    # 返回一个错误处理函数
    base::stop("Error: Unsupported distribution name: ", func_name)
  )
  
  # 3. 绑定参数环境
  base::environment(post_func) <- fn_env
  
  # 4. 返回生成的概率密度函数
  return(post_func)
}
