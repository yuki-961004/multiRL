#' @title TD Model
#' @description
#' 
#'  Learning Rate: \eqn{\alpha}
#'  
#'  \deqn{Q_{new} = Q_{old} + \alpha \cdot (R - Q_{old})}
#'  
#'  Inverse Temperature: \eqn{\beta}
#'  
#'  \deqn{
#'    P_{t}(a) = 
#'    \frac{
#'      \exp(\beta \cdot Q_{t}(a))
#'    }{
#'      \sum_{i=1}^{k} \exp(\beta \cdot Q_{t}(a_{i}))
#'    }
#'  }
#'  
#' @param params 
#'  Parameters used by the model’s internal functions,
#'    see \link[multiRL]{params}
#' 
#' @section Body: 
#' \preformatted{TD <- function(params){
#'   
#'   params <- list(
#'     free = list(alpha = params[1], beta = params[2])
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
