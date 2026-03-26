.name_rnnouts <- function(X_pred, loss, param_names) {
  switch(
    EXPR = loss,
    "MSE" = ,
    "MAE" = ,
    "HBR" = {
      colnames(X_pred) <- param_names
    },
    "NLL" = {
      n_params <- length(param_names)
      idx <- (n_params + 1):(2 * n_params)

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
    "MDN" = {
      K <- 3 # 混合成分数量
      n_params <- length(param_names)

      # 准备一个矩阵来存储点估计值
      point_estimates <- matrix(NA, nrow = nrow(X_pred), ncol = n_params)

      # 定义各部分的索引范围
      idx_pi_logits_all <- 1:(n_params * K)
      idx_mu_all <- (n_params * K + 1):(2 * n_params * K)
      idx_log_var_all <- (2 * n_params * K + 1):(3 * n_params * K)

      # 1. 计算点估计 (E[X] = sum(pi * mu)) 并转换 pi_logits
      for (i in 1:n_params) {
        # 当前参数对应的 pi_logits 和 mu 的列索引
        pi_cols <- idx_pi_logits_all[((i - 1) * K + 1):(i * K)]
        mu_cols <- idx_mu_all[((i - 1) * K + 1):(i * K)]

        # 提取 logits 和 mu
        logits <- X_pred[, pi_cols, drop = FALSE]
        mu_values <- X_pred[, mu_cols, drop = FALSE]

        # 计算 pi (softmax), for numerical stability
        exp_logits <- exp(logits - apply(logits, 1, max))
        pi_values <- exp_logits / rowSums(exp_logits)

        # 计算点估计 (加权平均)
        point_estimates[, i] <- rowSums(pi_values * mu_values)

        # 将原始 X_pred 中的 pi_logits 替换为真实的 pi 概率
        X_pred[, pi_cols] <- pi_values
      }

      # 2. 转换对数方差为标准差
      X_pred[, idx_log_var_all] <- sqrt(exp(X_pred[, idx_log_var_all]))

      # 3. 准备所有列的名称
      colnames(point_estimates) <- param_names
      name_pi <- paste0(
        "pi",
        rep(1:K, times = n_params),
        "_",
        rep(param_names, each = K)
      )
      name_mu <- paste0(
        "mu",
        rep(1:K, times = n_params),
        "_",
        rep(param_names, each = K)
      )
      name_sd <- paste0(
        "sd",
        rep(1:K, times = n_params),
        "_",
        rep(param_names, each = K)
      )

      # 4. 合并点估计和完整的分布信息
      colnames(X_pred) <- c(name_pi, name_mu, name_sd)
      X_pred <- cbind(point_estimates, X_pred)
    },
    stop("Unsupported loss type.")
  )

  return(X_pred)
}
