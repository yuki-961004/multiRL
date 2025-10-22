#' Utility Model
#'
#' @param params params
#'
#' @returns result
#' 
Utility <- function(params){
  
  params <- list(
    free = list(
      alpha = params[1], beta = params[2], gamma = params[3]
    ),
    fixed = list(
      delta = 0.1, epsilon = NA_real_, zeta = 1, eta = NA_real_
    ),
    constant = list(
      Q1 = NA_real_, lapse = 0.01
    )
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
