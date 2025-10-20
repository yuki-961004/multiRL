.block_ratio <- function(data, colname, levels) {
  
  target_factor <- base::factor(x = data[[colname]], levels = levels)
  target_table <- base::table(target_factor)
  ratio <- target_table / base::sum(target_table)
  
  return(ratio)
}