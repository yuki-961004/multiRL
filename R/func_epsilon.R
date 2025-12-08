#' Function: Epsilon Related
#'
#' @param rownum rownum
#' @param params params
#' @param ... extra
#'
#' @returns try
#' 
func_epsilon <- function(
    rownum,
    params,
    ...
){
  # if you need extra information
  # e.g.
  # Trial <- idinfo["Trial"]
  # Frame <- exinfo["Frame"]
  
  epsilon   <-  get_param(params, "epsilon")
  threshold <-  get_param(params, "threshold")
  
  if (is.na(epsilon) && threshold > 0) {
    model <- "first"
  } else if (!(is.na(epsilon)) && threshold == 0) {
    model <- "decreasing"
  } else if (!(is.na(epsilon)) && threshold == 1) {
    model <- "greedy"
  } else {
    stop("Unknown Model! Plase modify your learning rate function")
  }
  
  set.seed(rownum)
  # Epsilon-First: 
  if (rownum <= threshold) {
    try <- 1
  } else if (rownum > threshold && model == "first") {
    try <- 0
  # Epsilon-Greedy:
  } else if (rownum > threshold && model == "greedy"){
    try <- sample(
      c(1, 0),
      prob = c(epsilon, 1 - epsilon),
      size = 1
    )
  # Epsilon-Decreasing: 
  } else if (rownum > threshold && model == "decreasing") {
    try <- sample(
      c(1, 0),
      prob = c(
        1 / (1 + epsilon * rownum),
        epsilon * rownum / (1 + epsilon * rownum)
      ),
      size = 1
    )
  }
  
  return(try)
}
