func_beta <- function(
    x, 
    beta,
    ...
){
  
  if(!(is.numeric(x))) {
    stop("x must be a numeric vector")
  }
  
  if (beta == 0) {
    prob <- rep(1 / length(x), length(x))
  }
  else {
    exp_stable <- exp(beta * (x - max(x, na.rm = TRUE)))
    
    prob <- exp_stable / sum(exp_stable, na.rm = TRUE)
  }

  return(prob)
}
