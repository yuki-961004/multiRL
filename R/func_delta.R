func_delta <- function(
    count,
    params,
    ...
){
  delta     <-  get_param(params, "delta")

  bias <- delta * sqrt(log(count + exp(1)) / (count + 1e-10))
  
  return(bias)
}
