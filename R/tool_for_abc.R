.for_abc <- function(data, rsp, block, action) {
  
  n_block <- length(unique(data[[block]]))
  n_rsp <- length(rsp)
  
  # 计算每个block中simulation的选择比率
  ratio <- lapply(X = split(data, data[, block]), FUN = function(x) {
    action_prop  <- .block_ratio(data = x, colname = action, levels = rsp)
  })
  ratio_mat <- do.call(what = rbind, args = ratio)
  # 如果信息量过大, 则会进行PCA降维
  if ((n_block * n_rsp) > 50) {
    onerow <- t(stats::prcomp(x = ratio_mat, center = TRUE, rank. = 1)$x[, 1])
  } else {
    # 如果信息量不大, 则直接拍扁成单行
    onerow <- matrix(data = t(ratio_mat), nrow = 1)
  }
  
  return(list(ratio = ratio, onerow = onerow))
}
