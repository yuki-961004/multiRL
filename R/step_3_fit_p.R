#' @title 
#' Step 3: Optimizing parameters to fit real data
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
#' @section Example: 
#' \preformatted{ # fitting
#'  fitting.MLE <- multiRL::fit_p(
#'    estimate = "MLE",
#'   
#'    data = multiRL::TAB,
#'    colnames = list(
#'      object = c("L_choice", "R_choice"), 
#'      reward = c("L_reward", "R_reward"),
#'      action = "Sub_Choose"
#'    ),
#'    behrule = list(
#'      cue = c("A", "B", "C", "D"),
#'      rsp = c("A", "B", "C", "D")
#'    ),
#'   
#'    models = list(multiRL::TD, multiRL::RSTD, multiRL::Utility),
#'    settings = list(list(name = "TD"), list(name = "RSTD"), list(name = "Utility")),
#'   
#'    algorithm = "NLOPT_GN_MLSL",
#'    lowers = list(c(0, 0), c(0, 0, 0), c(0, 0, 0)),
#'    uppers = list(c(1, 5), c(1, 1, 5), c(1, 5, 1)),
#'    control = list(core = 10, iter = 100)
#'  )
#' }
fit_p <- function(
    estimate,
    
    data, 
    colnames,
    behrule,
    ids = NULL,
    
    funcs = NULL,
    priors = NULL,
    settings = NULL,
    models,
    
    algorithm,
    lowers,
    uppers,
    control,
    ...
){
################################# [check] ######################################
  
  if (estimate == "RNN") {.check_tensorflow()}
  
################################ [default] #####################################
  
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
  
  # 默认方程
  if (is.null(funcs)) {funcs <- rep(list(list()), length(models))}
  for (i in 1:length(funcs)) {
    default <- list(
      rate_func = multiRL::func_alpha,
      prob_func = multiRL::func_beta,
      util_func = multiRL::func_gamma,
      bias_func = multiRL::func_delta,
      expl_func = multiRL::func_epsilon,
      dcay_func = multiRL::func_zeta
    )
    funcs[[i]] <- utils::modifyList(x = default, val = funcs[[i]])
  }
  
  # 默认先验
  if (is.null(priors)) {
    priors <- rep(list(list()), length(models))
  } else {
    fit_priors <- .convert_priors(priors = priors, to = "dfunc")
  }
  
  # 默认设置
  if (is.null(settings)) {settings <- rep(list(list()), length(models))}
  for (i in 1:length(settings)) {
    default <- list(
      name = paste0("Unknown_", i)
    )
    settings[[i]] <- utils::modifyList(x = default, val = settings[[i]])
  }
  
  fit_settings <- rep(list(list()), length(models))
  for (i in 1:length(fit_settings)) {
    fit_settings[[i]] <- list(
      name = settings[[i]]$name,
      mode = "fitting",
      estimate = estimate
    )
  }
  
  # 默认控制
  default = list(
    # LBI
    iter = 10,
    pars = NA,
    size = 50,
    seed = 123,
    core = 1,
    diff = 0.001,
    # SBI
    sample = 100,
    scope = "individual",
    layer = "GRU",
    info = c(colnames$object, colnames$action),
    units = 128,
    batch_size = 10,
    epochs = 100
  )
  control <- utils::modifyList(x = default, val = control)
  
  # 解放control中的设定, 变成全局变量
  list2env(control, envir = environment())
  
  ############################ [aotu-detect data] ################################
  
  suppressMessages({dfinfo <- .detect_data(data)})
  # 如果没有输入被试序号的列名. 则自动探测
  if ("subid" %in% names(colnames)) {
    subid <- colnames[["subid"]]
  } else {
    subid <- dfinfo$sub_col_name
  }
  # 如果没有输入要拟合的被试序号, 就拟合所有的
  if (is.null(ids)){ids <- dfinfo$all_ids}
  
############################### [results] ###################################### 
  
  fitting <- estimation_methods(
    data = data, 
    behrule = behrule,
    ids = ids,
    colnames = colnames,
    funcs = funcs,
    priors = fit_priors,
    settings = fit_settings,
    models = models,
    estimate = estimate,
    algorithm = algorithm,
    lowers = lowers,
    uppers = uppers,
    control = control
  )
  
  fitting$fit_model <- factor(
    fitting$fit_model, levels = unique(fitting$fit_model)
  )
  
  fitting <- split(fitting, fitting$fit_model)
  
  class(fitting) <- "multiRL.fitting"
  
  return(fitting)
}