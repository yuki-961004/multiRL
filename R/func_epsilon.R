#' @title Function: \eqn{\epsilon}–first, Greedy, Decreasing 
#' @description
#' 
#'  \eqn{\epsilon–first}: 
#'  
#'  \deqn{
#'  P(x) = 
#'  \begin{cases}
#'    i \le \text{threshold}, & x=1  \\
#'    i > \text{threshold}, & x=0 
#'  \end{cases}
#'  }
#'  
#'  \eqn{\epsilon–greedy}: 
#'  
#'  \deqn{
#'  P(x) = 
#'  \begin{cases}
#'    \epsilon, & x=1  \\
#'    1-\epsilon, & x=0 
#'  \end{cases}
#'  }
#'  
#'  \eqn{\epsilon–decreasing}:
#'  
#'  \deqn{
#'  P(x) = 
#'  \begin{cases}
#'    \frac{1}{1+\epsilon \cdot i}, & x=1  \\
#'    \frac{\epsilon \cdot i}{1+\epsilon \cdot i}, & x=0 
#'  \end{cases}
#'  }
#' 
#' @param rownum 
#'  The trial number
#' @param params 
#'  Parameters used by the model’s internal functions,
#'    see \link[multiRL]{params}
#' @param ... 
#'  Subject ID, Block ID, Trial ID, and any additional information defined by 
#'    the user.
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
