#' estimate_1_MLE
#'
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
#' @returns data.frame
#' 
estimate_1_MLE <- function(
    data, 
    colnames,
    behrule,
    ids = NULL,
    
    models,
    funcs = NULL,
    priors,
    settings = NULL,
    
    algorithm,
    lowers,
    uppers,
    control,
    ...
){

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
    priors <- .convert_priors(priors = priors, to = "dfunc")
  }
  
  # 默认设置
  settings <- .restructure_settings(x = settings, n = length(models))
  for (i in 1:length(settings)) {
    default <- list(
      name = paste0("Unknown_", i),
      policy = "off"
    )
    settings[[i]] <- utils::modifyList(x = default, val = settings[[i]])
  }
  
  # 强制设置
  for (i in 1:length(settings)) {
    settings[[i]]$mode <- "fitting"
    settings[[i]]$estimate <- "MLE"
  }
  
  # 默认控制
  default = list(
    pars = NA,
    size = 50,
    iter = 10,
    seed = 123,
    core = 1
  )
  control <- utils::modifyList(x = default, val = control)
  
  # 解放control中的设定, 变成全局变量
  list2env(control, envir = environment())
  
  # 读取MLE迭代次数
  if (length(iter) == 1) {
    iter <- iter
  } else if (length(iter) == 2) {
    iter <- iter[1]
  }
  
############################ [aotu-detect data] ################################
  
  # 自动探测数据
  suppressMessages({dfinfo <- .detect_data(data)})
  # 如果没有输入被试序号的列名. 则自动探测
  if ("subid" %in% names(colnames)) {
    subid <- colnames[["subid"]]
  } else {
    subid <- dfinfo$sub_col_name
  }
  
  # 如果没有输入要拟合的被试序号, 就拟合所有的
  if (is.null(ids)){ids <- dfinfo$all_ids}
  
################################ [ Parallel ] ################################## 
  
  sys <- Sys.info()[["sysname"]]
  
  if (core == 1) {
    future::plan(future::sequential)
  } else if (sys == "Windows") {
    future::plan(future::multisession, workers = core)
  } else if (sys == "Darwin") {
    future::plan(future::multisession, workers = core)
  } else if (sys == "Linux") {
    future::plan(future::multicore, workers = core)
  }
  
  doFuture::registerDoFuture()
  
################################ [ beforeach ] ################################# 
  
  # 创建空list, 用于存放结果
  multiRL.models <- rep(list(list()), length(models))
  # 定义foreach内的指针
  j <- NA
  
  for (i in 1:length(models)) {

    model_name <- settings[[i]]$name
    
################################### [ MLE ] ####################################
    
    message(paste0(
      "Fitting ", model_name,"\n"
    ))
    
    # 定义进度条
    progressr::handlers(progressr::handler_txtprogressbar)
    # 进度条启动
    progressr::with_progress({
      # 进度条参照
      p <- progressr::progressor(steps = length(ids))
      # 锁定并行内种子
      doRNG::registerDoRNG(seed = seed)
      # MLE并行开始
      suppressMessages({
        multiRL.models[[i]] <- foreach::foreach(
          j = ids, .packages = c("multiRL")
        ) %dorng% {
          env <- estimate_0_ENV(
            data = data[data[, subid] == j, ],
            behrule = behrule,
            colnames = colnames,
            funcs = funcs[[i]],
            priors = priors[[i]],
            settings = settings[[i]]
          )
          out <- estimate_1_LBI(
            model = models[[i]],
            env = env,
            algorithm = algorithm,
            lower = lowers[[i]],
            upper = uppers[[i]],
            control = control
          )
          p()
          return(out)
        }
      })
    })
  }
  # 停止并行
  future::plan(future::sequential)
  
################################### [ END ] ####################################
  
  # 整理结果成表格
  result <- .extract_results(multiRL.models)
  
  return(result)
}
