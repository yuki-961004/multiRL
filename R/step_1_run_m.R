#' @title 
#' Step 1: Building reinforcement learning model
#'
#' @param data data
#' @param behrule behrule
#' @param colnames colnames
#' @param params params
#' @param funcs funcs
#' @param priors priors
#' @param settings settings
#' @param ... extra
#' 
#' @returns multiRL.model
#' 
run_m <- function(
    data,
    behrule,
    colnames,
    params,
    funcs = list(
      rate_func = multiRL::func_alpha,
      prob_func = multiRL::func_beta,
      util_func = multiRL::func_gamma,
      bias_func = multiRL::func_delta,
      expl_func = multiRL::func_epsilon
    ),
    priors = list(),
    settings = list(
      mode = "fit",
      policy = "on"
    ),
    ...
){
  
  multiRL.input <- process_1_input(
    data = data,
    colnames = colnames,
    params = params,
    funcs = funcs,
    priors = priors,
    settings = settings,
    ...
  )
  
  multiRL.behrule <- process_2_behrule(
    behrule = behrule,
    ...
  )
  
  multiRL.record <- process_3_record(
    input = multiRL.input,
    behrule = multiRL.behrule,
    ...
  )
  
  multiRL.output <- process_4_output(
    record = multiRL.record,
    ...
  )
  
  multiRL.metric <- process_5_metric(
    output = multiRL.output,
    ...
  )
  
  multiRL.model <- methods::new(
    Class = "multiRL.model", 
    multiRL.metric
  )
  
  return(multiRL.model)
}
