#' estimation_methods
#'
#' @param estimate 
#'  Estimate method that you want to use, 
#'    see \link[multiRL]{estimate}
#' @param data 
#'  A data frame in which each row represents a single trial,
#'    see \link[multiRL]{data} 
#' @param colnames 
#'  Column names in the data frame,
#'    see \link[multiRL]{colnames}
#' @param behrule 
#'  The agent’s implicitly formed internal rule,
#'    see \link[multiRL]{behrule}
#' @param ids 
#'  The Subject ID of the participant whose data needs to be fitted.
#' @param models 
#'  Reinforcement Learning Models
#' @param funcs 
#'  The functions forming the reinforcement learning model,
#'    see \link[multiRL]{funcs}
#' @param priors 
#'  Prior probability density function of the free parameters,
#'    see \link[multiRL]{priors}
#' @param settings 
#'  Other model settings, 
#'    see \link[multiRL]{settings}
#' @param algorithm 
#'  Algorithm packages that multiRL supports, 
#'    see \link[multiRL]{algorithm}
#' @param lowers 
#'  Lower bound of free parameters in each model.
#' @param uppers 
#'  Upper bound of free parameters in each model.
#' @param control 
#'  Settings manage various aspects of the iterative process,
#'    see \link[multiRL]{control}
#' @param ... 
#'  Additional arguments passed to internal functions.
#'
#' @returns results
#' 
estimation_methods <- function(
    estimate,
  
    data, 
    colnames,
    behrule,
    ids = NULL,
    
    models,
    funcs = NULL,
    priors = NULL,
    settings = NULL,
    
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
        
        models = models,
        funcs = funcs,
        priors = priors,
        settings = settings,
        
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
        
        models = models,
        funcs = funcs,
        priors = priors,
        settings = settings,

        algorithm = algorithm,
        lowers = lowers,
        uppers = uppers,
        control = control
      )
    },
    "ABC" = {
      estimate_2_ABC(
        data = data,
        behrule = behrule,
        ids = ids,
        colnames = colnames,
        
        models = models,
        funcs = funcs,
        priors = priors,
        settings = settings,
        
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
        
        models = models,
        funcs = funcs,
        priors = priors,
        settings = settings,
        
        control = control
      )
    },
  )
  
  return(result)
}