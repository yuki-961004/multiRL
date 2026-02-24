.reduce_sumstats <- function(method, abc, target) {

  # 读取基本信息
  n_params <- ncol(abc$df_params)
  # 不修改原始变量
  abc_reduction <- abc
  target_reduction <- target
  
  # 如果没有设定降维方法, 则不降维, 使用原始数据
  if (method == "NONE") {
    
    df_sumstats <- as.data.frame(do.call(
      rbind,
      lapply(abc_reduction$list_sumstats, function(ratio_mat) {
        as.vector(t(ratio_mat))
      })
    ))
    
    abc_reduction$df_sumstats <- df_sumstats
    
    target_reduction$onerow <- target$onerow
  
  # 如果降维方法是PLS, 则降维到一维, 且信息量和参数数量相等
  } else if (method == "PLS") {
    df_sumstats <- as.data.frame(do.call(
      rbind,
      lapply(abc$list_sumstats, function(ratio_mat) {
        as.vector(t(ratio_mat))
      })
    ))
    
    abc_reduction$df_sumstats <- df_sumstats
    
    pls_model <- pls::plsr(
      as.matrix(abc$df_params) ~ as.matrix(abc_reduction$df_sumstats),
      ncomp = n_params,
      method = "oscorespls"
    )
    
    abc_reduction$df_sumstats <- as.data.frame(
      pls::scores(pls_model)[, 1:n_params]
    )
    
    target_reduction$onerow <- t(matrix(stats::predict(
      pls_model, newdata = target$onerow, comps = 1:n_params
    )))
    
  # 如果降维方法是PCA, 则降维到一维, 且信息量和block数相等.   
  } else if (method == "PCA") {
    
    df_sumstats <- as.data.frame(do.call(
      rbind,
      lapply(abc_reduction$list_sumstats, function(ratio_mat) {
        t(stats::prcomp(x = ratio_mat, center = TRUE, rank. = 1)$x[, 1])
      })
    ))
    
    abc_reduction$df_sumstats <- df_sumstats
    
    target_reduction$onerow <- t(
      stats::prcomp(x = target_reduction$ratio, center = TRUE, rank. = 1)$x[, 1]
    )
    
  }
  
  reduction <- list(
    abc = abc_reduction,
    target = target_reduction
  )
  
  return(reduction)
}
