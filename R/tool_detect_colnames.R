.detect_colnames <- function(data, prefix) {
  
  # 检测以Object_和Reward_开头的列名
  colnames <- grep(
    pattern = paste0("^", prefix),
    x = names(data),
    value = TRUE
  )
  
  # nocov start
  if (length(colnames) == 0) {
    stop(
      "Could not automatically detect columns with prefix '",
      prefix,
      "'. Please manually specify column names."
    )
  }
  # nocov end
  
  return(colnames)
}
