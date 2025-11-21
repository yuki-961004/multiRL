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
#' @param engine engine
#' @param ... extra
#' 
#' @returns multiRL.model
#' 
run_m <- function(
    data,
    behrule = list(),
    colnames = list(),
    params = list(),
    funcs = list(),
    priors = list(),
    settings = list(),
    engine = "Cpp",
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
  
  if (engine == "Cpp") {
    multiRL.output <- process_4_output_cpp(
      record = multiRL.record,
      extra = list(...)
    )
  } else if (engine == "R") {
    multiRL.output <- process_4_output_r(
      record = multiRL.record,
      ...
    )
  }
  
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
