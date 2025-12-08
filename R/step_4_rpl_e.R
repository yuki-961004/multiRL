#' @title 
#' Step 4: Replaying the experiment with optimal parameters
#'
#' @param data data
#' @param behrule behrule
#' @param ids ids
#' @param colnames colnames
#' @param funcs funcs
#' @param priors priors
#' @param settings settings
#' @param result result
#' @param models models
#' @param free_params free_params
#' @param ... extra
#'
#' @returns replay
#' 
rpl_e <- function(
    data, 
    behrule,
    
    ids = NULL,
    colnames,
    funcs = NULL,
    priors = NULL,
    settings = NULL,
    
    result,
    models,
    free_params = NULL,
    ...
){
 
################################## [extra] #####################################
  
  extra <- list(...)
  omit <- extra$omit %||% base::character()
 
################################## [check] #####################################
   
  if (inherits(result, "multiRL.recovery")) {
    mode <- "recovery"
  } else if (inherits(result, "multiRL.fitting")) {
    mode <- "fitting"
  } else {
    stop("The provided 'result' does not belong to the 'multiRL' class.")
  }
  
################################# [replay] #####################################
  
  if (mode == "recovery") {
    
    simulate <- .replay(
      data = data, 
      behrule = behrule,
      ids = ids,
      colnames = colnames,
      funcs = funcs,
      priors = priors,
      settings = settings,
      result = .rbind_fill(result$simulate),
      models = models,
      free_params = free_params,
    )
    
    model_names <- names(result$recovery)
    
    recovery <- list()
    
    for (name in model_names) {
      recovery[[name]] <- .replay(
        data = data, 
        behrule = behrule,
        ids = ids,
        colnames = colnames,
        funcs = funcs,
        priors = priors,
        settings = settings,
        result = result$recovery[[name]],
        models = models,
        free_params = free_params,
      )
    }
    
    replay <- list(
      simulate = simulate,
      recovery = recovery
    )
  } else if (mode == "fitting") {
    fitting <- .replay(
      data = data, 
      behrule = behrule,
      ids = ids,
      colnames = colnames,
      funcs = funcs,
      priors = priors,
      settings = settings,
      result = .rbind_fill(result),
      models = models,
      free_params = free_params,
    )
    
    replay <- fitting
  }
  
  replay <- .remove_slot(replay, omit = omit)
  
  class(replay) <- "multiRL.replay"
  
  return(replay)
}