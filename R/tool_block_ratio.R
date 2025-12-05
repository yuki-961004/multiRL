.block_ratio <- function(data, colname, levels) {
  
  # 计算ABC所需的摘要统计量: 每个block中latent和simulation的比率
  counts <- tabulate(match(data[[colname]], levels), length(levels))
  ratio <- counts / sum(counts)
  
  return(ratio)
}
