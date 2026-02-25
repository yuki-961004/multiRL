.reduce_sumstats <- function(abc, target, method, ncomp) {

  # 读取基本信息
  if (is.null(ncomp)) {
    ncomp <- nrow(target$ratio)
  }

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
      ncomp = ncomp,
      method = "simpls",
      center = TRUE,
      scale. = TRUE
    )
    
    abc_reduction$df_sumstats <- as.data.frame(
      pls::scores(pls_model)[, 1:ncomp]
    )
    
    target_reduction$onerow <- as.matrix(
      stats::predict(
        pls_model,
        newdata = target$onerow,
        type = "scores",
        comps = 1:ncomp
      )
    )
    
  # 如果降维方法是PCA, 则降维到一维, 且信息量和block数相等.   
  } else if (method == "PCA") {
    
    df_sumstats <- as.data.frame(do.call(
      rbind,
      lapply(abc_reduction$list_sumstats, function(ratio_mat) {
        as.vector(t(ratio_mat))
      })
    ))
    
    colnames(df_sumstats) <- NULL
    
    pca_model <- stats::prcomp(
      df_sumstats,
      center = TRUE,
      scale. = TRUE
    )
    
    abc_reduction$df_sumstats <- as.data.frame(
      pca_model$x[, 1:ncomp]
    )
    
    target_reduction$onerow <- t(as.matrix(
      stats::predict(pca_model, newdata = target$onerow)[, 1:ncomp]
    ))
  }
  
  reduction <- list(
    abc = abc_reduction,
    target = target_reduction
  )
  
  return(reduction)
}
