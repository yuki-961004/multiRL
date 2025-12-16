#' RSTD Model
#'
#'  \deqn{Q_{new} = Q_{old} + \alpha_{-} \cdot (R - Q_{old}), R < Q_{old}}
#'  \deqn{Q_{new} = Q_{old} + \alpha_{+} \cdot (R - Q_{old}), R \ge Q_{old}}
#'
#' @param params 
#'  Parameters used by the model’s internal functions,
#'    see \link[multiRL]{params}
#' 
RSTD <- function(params){
  
  params <- list(
    free = list(alphaN = params[1], alphaP = params[2], beta = params[3])
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
