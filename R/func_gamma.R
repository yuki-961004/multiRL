func_gamma <- function(
    x,
    gamma = 1,
    ...
){
  if(!(is.numeric(x))) {
    stop("x must be a numeric vector")
  }
  
  utility <- x^gamma
}
