.block_ratio <- function(data, colname, levels) {
  
  # 计算ABC所需的摘要统计量: 每个block中latent和simulation的比率
  target_factor <- factor(x = data[[colname]], levels = levels)
  target_table <- table(target_factor)
  ratio <- target_table / sum(target_table)
  
  return(ratio)
}