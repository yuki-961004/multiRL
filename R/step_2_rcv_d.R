#' @title 
#' Step 2: Generating fake data for parameter and model recovery
#'
#' @param data data
#' @param behrule behrule
#' @param id id
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
#' @returns recovery
#' 
rcv_d <- function(
  data, 
  behrule,
  
  id = NULL,
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
    sim_priors <- .convert_priors(priors = priors, to = "rfunc")
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
  
  sim_settings <- rep(list(list()), length(models))
  for (i in 1:length(sim_settings)) {
    sim_settings[[i]] <- list(
      name = settings[[i]]$name,
      mode = "simulating",
      estimate = estimate,
      policy = "on"
    )
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
    train = 1000,
    scope = "shared",
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
  # 只需要用一个被试的模板生成模拟数据
  if (is.null(id)){id <- dfinfo$random_id}
  # 在整体数据集中索引这个随机被试的数据
  data = data[data[, subid] == id, ]
  
############################## [simulate data] #################################
  
  simulated_data <- list()
  simulated_params <- list()
  list_recovery <- list()
  
  for (i in 1:length(models)) {
    
    model_name <- settings[[i]]$name
    
    message(paste0(
      "Recovery ", model_name,"\n"
    ))
    
    multiRL.env <- estimate_0_ENV(
      data = data,
      behrule = behrule,
      colnames = colnames,
      settings = sim_settings[[i]],
    )
    
    list_simulated <- estimate_2_SBI(
      model = models[[i]],
      env = multiRL.env,
      priors = sim_priors[[i]],
      control = control
    )  
    
    simulated_data[[i]] <- .list2df(list = list_simulated, subid = subid)[[1]]
    simulated_params[[i]] <- .list2df(list = list_simulated, subid = subid)[[2]]
    simulated_params[[i]]$simulate_model <- model_name
    names(simulated_params)[i] <- model_name
    
############################## [recovery data] #################################
    
    list_recovery[[i]] <- estimation_methods(
      data = simulated_data[[i]], 
      behrule = behrule,
      ids = NULL,
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
    
    names(list_recovery)[i] <- model_name
  }
  
############################### [results] ###################################### 
  
  result <- list(
    simulate = simulated_params,
    recovery = list_recovery
  )
  
  return(result)

}
