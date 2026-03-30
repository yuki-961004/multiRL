#' @title 
#' Tool for Generating an Environment for Models
#' @name estimate_0_ENV
#' @description 
#'  This function creates an independent R environment for each model (or 
#'    object function) when searching for optimal parameters using an algorithm 
#'    package. Such isolation is especially important when parameter 
#'    optimization is performed in parallel across multiple subjects. The 
#'    function transfers standardized input parameters into a dedicated 
#'    environment, ensuring that each model is evaluated in a self-contained 
#'    and interference-free context.
#'  
#' @param data 
#'  A data frame in which each row represents a single trial,
#'    see \link[multiRL]{data} 
#' @param colnames 
#'  Column names in the data frame,
#'    see \link[multiRL]{colnames}
#' @param behrule 
#'  The agent's implicitly formed internal rule,
#'    see \link[multiRL]{behrule}
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
#' @returns An \code{environment}, \code{multiRL.env} contains all variables 
#'    required by the objective function and is used to isolate environments 
#'    during parallel computation. 
#' 
estimate_0_ENV <- function(
    data,
    colnames = list(),
    behrule,
    funcs = list(),
    priors = list(),
    settings = list(),
    ...
){
  
################################# [default] ####################################
  
  # 默认列名
  default <- list(
    subid = "Subject", 
    block = "Block", 
    trial = "Trial",
    object = NA_character_, 
    reward = NA_character_, 
    action = "Action"
  )
  colnames <- utils::modifyList(x = default, val = colnames)
  
  # 默认函数
  default <- list(
    lrng_func = multiRL::func_alpha,
    prob_func = multiRL::func_beta,
    util_func = multiRL::func_gamma,
    bias_func = multiRL::func_delta,
    expl_func = multiRL::func_epsilon,
    dcay_func = multiRL::func_zeta
  )
  funcs <- utils::modifyList(x = default, val = funcs)
  
  # 默认设置
  default <- list(
    name = "Unknown",
    mode = "fitting",
    estimate = "MLE",
    policy = "on"
  )
  settings <- utils::modifyList(x = default, val = settings)
  
################################# [new.env] ####################################
  
  multiRL.env <- new.env()
  
  multiRL.env$data <- data
  multiRL.env$behrule <- behrule
  multiRL.env$colnames <- colnames
  multiRL.env$funcs <- funcs
  multiRL.env$priors <- priors
  multiRL.env$settings <- settings
  
  return(multiRL.env)
}
