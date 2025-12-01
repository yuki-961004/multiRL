#' estimation_methods
#'
#' @param data data
#' @param behrule behrule
#' @param ids ids
#' @param colnames colnames
#' @param funcs funcs
#' @param priors priors
#' @param settings settings
#' @param models models
#' @param estimate estimate
#' @param algorithm algorithm
#' @param lowers lowers
#' @param uppers uppers
#' @param control control
#' @param ... extra
#'
#' @returns results
#' 
estimation_methods <- function(
    data, 
    behrule,
    
    ids = NULL,
    colnames,
    funcs = NULL,
    priors = NULL,
    settings = NULL,
    models,
    
    estimate,
    algorithm,
    lowers,
    uppers,
    control,
    ...
){

  result <- switch(
    EXPR = estimate,
    "MLE" = {
      estimate_1_MLE(
        data = data,
        behrule = behrule,
        ids = ids,
        colnames = colnames,
        funcs = funcs,
        priors = priors,
        settings = settings,
        models = models,
        algorithm = algorithm,
        lowers = lowers,
        uppers = uppers,
        control = control
      )
    },
    "MAP" = {
      estimate_1_MAP(
        data = data,
        behrule = behrule,
        ids = ids,
        colnames = colnames,
        funcs = funcs,
        priors = priors,
        settings = settings,
        models = models,
        algorithm = algorithm,
        lowers = lowers,
        uppers = uppers,
        control = control
      )
    },
    "RNN" = {
      estimate_2_RNN(
        data = data,
        behrule = behrule,
        ids = ids,
        colnames = colnames,
        funcs = funcs,
        priors = priors,
        settings = settings,
        models = models,
        control = control
      )
    },
  )
  
  return(result)
}