.detect_colnames <- function(data, prefix) {
  
  # 1. 获取所有列名
  all_names <- names(data)
  
  # 2. 使用 startsWith 匹配前缀
  matched_cols <- all_names[startsWith(all_names, prefix)]
  
  # 3. 错误处理
  # nocov start
  if (length(matched_cols) == 0L) {
    stop(
      "Could not automatically detect columns with prefix '", prefix, 
      "'. Please manually specify column names."
    )
  }
  # nocov end
  return(matched_cols)
}