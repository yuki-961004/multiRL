.extract_params <- function(x) {
  param_vector <- unlist(x$params)      
  names(param_vector) <- names(x$params)  
  return(param_vector)
}

.extract_sumstats <- function(x) {
  stats_vector <- x$sumstat
  names(stats_vector) <- colnames(x$sumstat)
  return(stats_vector)
}
