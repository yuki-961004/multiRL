.block_ratio <- function(data, colname, levels) {
  
  target_factor <- factor(x = data[[colname]], levels = levels)
  target_table <- table(target_factor)
  ratio <- target_table / sum(target_table)
  
  return(ratio)
}