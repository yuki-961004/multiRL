.dealwith_nlopt <- function(algorithm) {
  
  if (length(algorithm > 1)) {
    global_opts <- algorithm[[1]]
    local_opts <- list(algorithm = algorithm[[2]], xtol_rel = 1e-8)
    algorithm <- "NLOPT"
  } else if (length(algorithm == 1)) {
    global_opts <- algorithm
    local_opts <- NULL
    algorithm <- "NLOPT"
  } 
  
  result <- list(
    algorithm = algorithm, 
    global_opts = global_opts, 
    local_opts = local_opts
  )
}
