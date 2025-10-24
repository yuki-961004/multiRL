#' RSTD Model
#'
#' @param params params
#'
#' @returns result
#' 
RSTD <- function(params){
  
  params <- list(
    free = list(alpha_n = params[1], alpha_p = params[2], beta = params[3])
  )
  
  multiRL.model <- multiRL::run_m(
    data = data,
    behrule = behrule,
    colnames = colnames,
    params = params,
    funcs = funcs,
    priors = priors,
    settings = settings
  )
  
  assign(x = "multiRL.model", value = multiRL.model, envir = multiRL.env)
  return(.return_result(multiRL.model))
}
