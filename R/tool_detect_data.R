.detect_data <- function(data) {
  
  # 1. 查找包含 "sub" 的列名 (忽略大小写)
  # grep(value = TRUE) 直接返回列名, 避免了 index 操作和中间变量
  candidates <- grep("sub", names(data), ignore.case = TRUE, value = TRUE)
  
  if (length(candidates) == 0L) {
    stop("Error: No column name containing 'sub' found.")
  }
  
  # 默认取第一个, 如果有多个则提示
  target_col <- candidates[1L]
  if (length(candidates) > 1L) {
    message(
      "Multiple 'sub' columns found: ", paste(candidates, collapse = ", "), 
      ". Using: '", target_col, "'"
    )
  } else {
    message("Detected subject column: '", target_col, "'")
  }
  
  # 2. 提取有效ID
  # 直接操作, 避免 all_ids_in_column 这种中间对象
  # 使用 unique 配合 na.omit, 即使数据量大也比先提取再处理快
  valid_ids <- unique(stats::na.omit(data[[target_col]]))
  
  if (length(valid_ids) == 0L) {
    stop("Error: No valid IDs found in column '", target_col, "'.")
  }
  
  first_id <- valid_ids[1L]
  
  # 3. 返回结果 (不做 as.numeric 转换)
  # 既然追求简洁, 不需要显式创建 res 对象再 return
  res <- list(
    sub_col_name = target_col,
    random_id    = first_id,
    all_ids      = valid_ids
  )

  return(res)
}