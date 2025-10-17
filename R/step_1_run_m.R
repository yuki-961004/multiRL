#' @title 
#' Step 1: Building reinforcement learning model
#'
#' @param data data
#' @param colnames colnames
#' @param params params
#' @param funcs funcs
#' @param behrule behrule
#' @param ... extra
#'
#' @returns multiRL.model
#' 
run_m <- function(
    data,
    colnames,
    params,
    funcs,
    behrule,
    ...
){
  
  multiRL.input <- process_1_input(
    data = data,
    colnames = colnames,
    params = params,
    funcs = funcs,
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
