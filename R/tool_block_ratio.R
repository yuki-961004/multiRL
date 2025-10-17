.block_ratio <- function(data, colname) {

  target_col <- base::table(data[[colname]])
  
  ratio <- target_col / base::sum(target_col)
  
  return(ratio)
}
