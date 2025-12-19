.restructure_settings <- function(x, n){
  
  slots <- c("name", "mode", "estimate", "policy")
  
  # 每个元素都没名字, 且第一个元素是list
  if (is.null(names(x)) && is.list(x[[1]])) {
    # 说明每个模型的设置都是由单独list传入
    settings <- x
  # 如果以混合形式输入, 则需要转换成分割形式
  } else if (any(slots %in% names(x))) {
    settings <- rep(list(list()), n)
    present_slots <- intersect(names(x), slots)
    for (s in present_slots) {
      val <- x[[s]]
      for (i in seq_len(n)) {
        # 如果长度为 1，始终取第一个元素；否则取第 i 个
        settings[[i]][[s]] <- if (length(val) == 1) val[[1]] else val[[i]]
      }
    }
  # 如果settings里没设置, 就传出一个空list
  } else if (is.null(x)) {
    settings <- rep(list(list()), n)
  # nocov start
  } else {
    stop("Invalid settings value.")
  }
  # nocov end
  
  return(settings)
}
