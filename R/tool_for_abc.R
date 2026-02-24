.for_abc <- function(data, rsp, block, action) {
  
  n_block <- length(unique(data[[block]]))
  n_rsp <- length(rsp)
  
  # 计算每个block中simulation的选择比率
  ratio <- lapply(X = split(data, data[, block]), FUN = function(x) {
    action_prop  <- .block_ratio(data = x, colname = action, levels = rsp)
  })
  
  ratio_mat <- do.call(what = rbind, args = ratio)
  onerow <- matrix(data = t(ratio_mat), nrow = 1)
  
  return(list(ratio = ratio_mat, onerow = onerow))
}
