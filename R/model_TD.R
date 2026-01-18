#' @title Temporal Differences Model
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
#' @return Depending on the \code{mode} and \code{estimate} defined in the 
#'    runtime environment, the corresponding outputs for different estimation 
#'    methods are produced, such as a single log-likelihood value or summary 
#'    statistics.
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
