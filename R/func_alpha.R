func_alpha <- function(
    qvalue,
    reward,
    alpha,
    ...
){
  
  update <- qvalue + alpha * (reward - qvalue)
  
  return(update)
}
