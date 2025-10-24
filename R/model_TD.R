#' TD Model
#'
#' @param params params
#'
#' @returns result
#' 
TD <- function(params){
  
  params <- list(
    free = list(alpha = params[1], beta = params[2])
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
