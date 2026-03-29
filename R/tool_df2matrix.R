.df2matrix <- function(df) {
  out_df <- list()
  
  all_values <- unlist(lapply(df, as.character))
  
  suppressWarnings(num_values <- as.numeric(all_values))
  
  # 提取非数字的字符串，并按照 "_" 拆分，获取所有基础词汇 (Sub-tokens)
  str_values <- all_values[is.na(num_values) & !is.na(all_values)]
  all_tokens <- unlist(strsplit(str_values, "_"))
  
  # 建立全局基础词汇映射表（字典），这样 "Right" 无论出现在哪，数字都一样
  uniq_tokens <- sort(unique(all_tokens))
  mapping <- stats::setNames(seq_along(uniq_tokens), uniq_tokens)
  
  for (col_name in names(df)) {
    col_data <- df[[col_name]]
    
    if (is.numeric(col_data)) {
      out_df[[col_name]] <- col_data
      next
    }
    
    col_char <- as.character(col_data)
    suppressWarnings(num_col <- as.numeric(col_char))
    
    if (!any(is.na(num_col) & !is.na(col_data))) {
      out_df[[col_name]] <- num_col
      next
    }
    
    # 按 "_" 拆分。如果是单独的词(如Action)，max_len为1；如果是组合词，会自动分列
    split_list <- strsplit(col_char, "_")
    max_len <- max(sapply(split_list, length), na.rm = TRUE)
    
    if (max_len <= 1) {
      out_df[[col_name]] <- as.numeric(mapping[col_char])
    } else {
      for (i in 1:max_len) {
        # 提取每一行的第 i 个元素，缺失补 NA
        sub_col <- sapply(split_list, function(x) {
          if (length(x) >= i) x[i] else NA_character_
        })
        out_df[[paste0(col_name, "_", i)]] <- as.numeric(mapping[sub_col])
      }
    }
  }
  
  return(as.matrix(as.data.frame(out_df)))
}
