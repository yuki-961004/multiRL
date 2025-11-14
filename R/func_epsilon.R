#' Epsilon Related
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
  
  epsilon   <-  .get_param(params, "epsilon")
  zeta      <-  .get_param(params, "zeta")
  eta       <-  .get_param(params, "eta")

  set.seed(rownum)
  # Epsilon-First: 
  if (rownum <= zeta) {
    try <- 1
  } else if (rownum > zeta & is.na(epsilon) & is.na(eta)) {
    try <- 0
  # Epsilon-Greedy:
  } else if (rownum > zeta & !(is.na(epsilon)) & is.na(eta)){
    try <- sample(
      c(1, 0),
      prob = c(epsilon, 1 - epsilon),
      size = 1
    )
  # Epsilon-Decreasing: 
  } else if (rownum > zeta & is.na(epsilon) & !(is.na(eta))) {
    try <- sample(
      c(1, 0),
      prob = c(
        1 / (1 + eta * rownum),
        eta * rownum / (1 + eta * rownum)
      ),
      size = 1
    )
  }
  
  return(try)
}
