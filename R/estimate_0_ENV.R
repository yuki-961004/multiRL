#' estimate_0_ENV
#'
#' @param data data
#' @param behrule behrule
#' @param colnames colnames
#' @param funcs funcs
#' @param priors priors
#' @param settings settings
#' @param ... extra
#'
#' @returns multiRL.env
#' 
estimate_0_ENV <- function(
    data,
    behrule,
    colnames,
    funcs,
    priors,
    settings,
    ...
){
  multiRL.env <- new.env()
  
  multiRL.env$data <- data
  multiRL.env$behrule <- behrule
  multiRL.env$colnames <- colnames
  multiRL.env$funcs <- funcs
  multiRL.env$priors <- priors
  multiRL.env$settings <- settings
  
  return(multiRL.env)
}