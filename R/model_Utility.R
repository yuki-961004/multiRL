#' Utility Model
#'
#'  \deqn{Q_{new} = Q_{old} + \alpha \cdot (U(R) - Q_{old})}
#'  \deqn{U(R) = {R}^{\gamma}}
#'
#' @param params 
#'  Parameters used by the model’s internal functions,
#'    see \link[multiRL]{params}
#'
#' @section Body: 
#' \preformatted{Utility <- function(params){
#'   
#'   params <- list(
#'     free = list(alpha = params[1], beta = params[2], gamma = params[3])
#'   )
#'   
#'   multiRL.model <- multiRL::run_m(
#'     data = data,
#'     behrule = behrule,
#'     colnames = colnames,
#'     params = params,
#'     funcs = funcs,
#'     priors = priors,
#'     settings = settings
#'   )
#'   
#'   assign(x = "multiRL.model", value = multiRL.model, envir = multiRL.env)
#'   return(.return_result(multiRL.model))
#' }
#' }
#' 
Utility <- function(params){
  
  params <- list(
    free = list(alpha = params[1], beta = params[2], gamma = params[3])
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
