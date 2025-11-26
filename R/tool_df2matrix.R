.df2matrix <- function(df) {
  # 因为RNN只接受全数字的矩阵, 所以需要将表格中的字符串按照唯一值转换成字符串

  # 提取所有列，转为字符后拼成一个向量
  all_values <- unlist(lapply(df, as.character))
  
  # 检查哪些可以转为数字
  suppressWarnings(num_values <- as.numeric(all_values))
  
  # 提取非数字的唯一字符串并按字母排序
  uniq_str <- sort(unique(all_values[is.na(num_values)]))
  
  # 建立全局映射表（字典）
  mapping <- stats::setNames(seq_along(uniq_str), uniq_str)
  
  # 对每一列进行转换
  df[] <- lapply(df, function(col) {
    # 如果本身是数字型，直接返回
    if (is.numeric(col)) return(col)
    
    # 转字符
    col_char <- as.character(col)
    
    # 尝试转换为数字
    suppressWarnings(num_col <- as.numeric(col_char))
    
    # 如果全是数字字符，直接返回数值
    if (!any(is.na(num_col))) return(num_col)
    
    # 否则用全局映射表替换
    col_char[col_char %in% names(mapping)] <- as.character(
      mapping[col_char[col_char %in% names(mapping)]]
    )
    
    # 再转为数值
    as.numeric(col_char)
  })
  
  df <- as.matrix(df)
  
  return(df)
}
