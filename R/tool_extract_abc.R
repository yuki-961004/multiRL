.extract_params <- function(x) {
  param_vector <- unlist(x$params)      
  names(param_vector) <- names(x$params)  
  return(param_vector)
}

.extract_sumstats <- function(x) {
  stats_matrix <- x$sumstat
  colnames(stats_matrix) <- colnames(x$sumstat)
  return(stats_matrix)
}
