#' @title 
#' Step 4: Replaying the experiment with optimal parameters
#' 
#' @param result 
#'  Result from \code{rcv_d} or \code{fit_p}
#' @param free_params 
#'  In order to prevent ambiguity regarding the free parameters, their names 
#'    can be explicitly defined by the user.
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
#' @param ... 
#'  Additional arguments passed to internal functions.
#' 
#' @section Example: 
#' \preformatted{ # info
#'  data = multiRL::TAB
#'  colnames = list(
#'    object = c("L_choice", "R_choice"), 
#'    reward = c("L_reward", "R_reward"),
#'    action = "Sub_Choose"
#'  )
#'  behrule = list(
#'    cue = c("A", "B", "C", "D"),
#'    rsp = c("A", "B", "C", "D")
#'  )
#'  
#'  replay.recovery <- multiRL::rpl_e(
#'    result = recovery.MLE,
#'   
#'    data = data,
#'    colnames = colnames,
#'    behrule = behrule,
#'   
#'    models = list(multiRL::TD, multiRL::RSTD, multiRL::Utility),
#'    settings = list(name = c("TD", "RSTD", "Utility")),
#'   
#'    omit = c("data", "funcs")
#'  )
#' 
#'  replay.fitting <- multiRL::rpl_e(
#'    result = fitting.MLE,
#'   
#'    data = data,
#'    colnames = colnames,
#'    behrule = behrule,
#'   
#'    models = list(multiRL::TD, multiRL::RSTD, multiRL::Utility),
#'    settings = list(name = c("TD", "RSTD", "Utility")),
#'   
#'    omit = c("funcs")
#'  )
#' }
rpl_e <- function(
    result,  
    free_params = NULL,
    
    data, 
    colnames,
    behrule,
    ids = NULL,
    
    models,
    funcs = NULL,
    priors = NULL,
    settings = NULL,
    
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
  
############################ [aotu-detect data] ################################
  
  suppressMessages({dfinfo <- .detect_data(data)})
  # 如果没有输入被试序号的列名. 则自动探测
  if ("subid" %in% names(colnames)) {
    subid <- colnames[["subid"]]
  } else {
    subid <- dfinfo$sub_col_name
  }
  
################################# [replay] #####################################
  
  if (mode == "recovery") {
    
    id <- dfinfo$random_id
    data <- data[data[, subid] == id, ]
    
    sim_data <- list()
    for (i in 1:nrow(result$simulate[[1]])) {
      sim_data[[i]] <- data
      sim_data[[i]][[subid]] <- i
    }
    
    sim_data <- do.call(rbind, sim_data)
    
    simulate <- .replay(
      data = sim_data, 
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
        data = sim_data, 
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