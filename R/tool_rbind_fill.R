.rbind_fill <- function(x) {
   
  # 1. 搜集所有可能出现的列名 (Union of all keys)
  # 无论 x 的元素是 data.frame 还是 list，names() 都能正确获取名字
  all_cols <- unique(unlist(lapply(x, names), use.names = FALSE))
  
  # 如果列表为空或没有任何列名，直接返回空数据框
  if (length(all_cols) == 0) {
    return(data.frame())
  }
  
  # 2. 标准化每个元素 (Standardize)
  # 这一步极其关键: 确保每个元素都变成拥有相同列序的 data.frame
  standardized_list <- lapply(x, function(item) {
    
    # [关键修复]: 如果元素是普通列表(list)，必须先转为单行 data.frame
    if (!is.data.frame(item)) {
      # as.data.frame 处理 list 时，会将 list 的每个元素作为一列
      # stringsAsFactors = FALSE 是必须的，防止字符变成因子
      item <- as.data.frame(item, stringsAsFactors = FALSE)
    }
    
    # 找出缺失列
    missing_cols <- setdiff(all_cols, names(item))
    
    # 补齐缺失列为 NA
    if (length(missing_cols) > 0) {
      item[missing_cols] <- NA
    }
    
    # 重排各列以保持对齐 (drop = FALSE 保证单列不退化为向量)
    return(item[, all_cols, drop = FALSE])
  })
  
  # 3. 合并
  # 现在所有元素都是列名一致的 data.frame 了，可以安全 rbind
  res <- do.call(rbind, standardized_list)
  
  # 清理行名
  rownames(res) <- NULL
  
  return(res)
}