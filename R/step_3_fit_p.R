#' fit_p
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
#' @returns optimal params
#' 
fit_p <- function(
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
      dcay_func = multiRL::func_theta
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
      name = paste0("Unknown_", i),
      policy = "on"
    )
    settings[[i]] <- utils::modifyList(x = default, val = settings[[i]])
  }
  
  fit_settings <- rep(list(list()), length(models))
  for (i in 1:length(fit_settings)) {
    fit_settings[[i]] <- list(
      name = settings[[i]]$name,
      mode = "fitting",
      estimate = estimate,
      policy = settings[[i]]$policy
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
  
  result <- estimation_methods(
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
  
  return(result)
}