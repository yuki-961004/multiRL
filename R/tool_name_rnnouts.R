.name_rnnouts <- function(X_pred, loss, param_names) {
  switch(
    EXPR = loss,
    "NLL" = {
      n_params <- length(param_names)
      idx <- (n_params + 1) : (2 * n_params)
      
      # 将对数方差列转换为标准差
      X_pred[, idx] <- sqrt(exp(X_pred[, idx]))
      
      # 命名矩阵列：参数名 + sd_参数名
      colnames(X_pred) <- c(param_names, paste0("sd_", param_names))
    },
    "QRL" = {
      # QRL (Quantile Regression Loss): 
      colnames(X_pred) <- c(
        paste0(param_names, "_05"),
        param_names,
        paste0(param_names, "_95")
      )
    },
    "MSE" = {
      colnames(X_pred) <- param_names
    },
    stop("Unsupported loss type.")
  )
  
  return(X_pred)
}