#' estimate_1_MAP
#'
#' @param data data
#' @param behrule behrule
#' 
#' @param ids ids
#' @param colnames colnames
#' @param funcs funcs
#' @param priors priors
#' @param settings settings
#' @param models models
#' 
#' @param algorithm algorithm
#' @param lowers lowers
#' @param uppers uppers
#' @param control control
#' 
#' @param ... extra
#'
#' @returns data.frame
#' 
estimate_1_MAP <- function(
    data, 
    behrule,
    
    ids = NULL,
    colnames,
    funcs = NULL,
    priors,
    settings = NULL,
    models,
    
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
      dcay_func = multiRL::func_theta
    )
    funcs[[i]] <- utils::modifyList(x = default, val = funcs[[i]])
  }
  
  # 默认先验
  priors <- .convert_priors(priors = priors, to = "dfunc")
  
  # 默认设置
  if (is.null(settings)) {settings <- rep(list(list()), length(models))}
  for (i in 1:length(settings)) {
    default <- list(
      name = paste0("Unknown_", i),
      mode = "fitting",
      estimate = "MAP",
      policy = "on"
    )
    settings[[i]] <- utils::modifyList(x = default, val = settings[[i]])
  }
  
  # 默认控制
  default = list(
    pars = NA,
    size = 50,
    iter = c(10, 10),
    seed = 123,
    core = 1,
    diff = 0.001
  )
  control <- utils::modifyList(x = default, val = control)
  # 解放control中的设定, 变成全局变量
  list2env(control, envir = environment())
  
  # 读取MAP迭代次数
  if (length(iter) == 1) {
    limit <- iter
  } else if (length(iter) == 2) {
    limit <- iter[2]
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
      "Initializing ", model_name,"\n"
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
        multiRL.model.MLE <- foreach::foreach(
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
    
######################### [ Initialize Posteriors ] ############################
    
    posteriors <- .update_priors(x = multiRL.model.MLE, priors = priors[[i]])
    sum_LogPo <- sum(sapply(multiRL.model.MLE, function(x) x@sumstat@LPo))
    delta_LogPo = 1
    LogPo <- sum_LogPo
    
    if (is.infinite(LogPo)) {
      LogPo <- 0
      warning("Infinite in Log-Posterior Probability")
    }
    message(paste0(
      "Starting Expectation-Maximization Algorithm", "\n",
      "Log-Posterior Probability: ", round(LogPo, 2)
    ))
    
################################### [ MAP ] #################################### 
    
    iter <- 0
    stuck <- 0
    
    # 当LogPo的变化值不小于diff, 或迭代次数未达到, 则不断执行
    while (abs(delta_LogPo) > diff) {
    
      # 定义进度条
      progressr::handlers(progressr::handler_txtprogressbar)
      # 进度条启动
      progressr::with_progress({
        # 进度条参照
        p <- progressr::progressor(steps = length(ids))
        # 锁定并行种子
        doRNG::registerDoRNG(seed = seed)
        # MAP并行开始
        suppressMessages({
          multiRL.model.MAP <- foreach::foreach(
            j = ids, .packages = c("multiRL")
          ) %dorng% {
            env <- estimate_0_ENV(
              data = data[data[, subid] == j, ],
              behrule = behrule,
              colnames = colnames,
              funcs = funcs[[i]],
              priors = posteriors,
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
    
########################### [ Update Posteriors ] ##############################
      
      posteriors <- .update_priors(x = multiRL.model.MAP, priors = posteriors)
      sum_LogPo <- sum(sapply(multiRL.model.MAP, function(x) x@sumstat@LPo))
      if (is.infinite(LogPo) || 
          is.infinite(sum_LogPo) || 
          delta_LogPo == LogPo - sum_LogPo) {
        stuck <- stuck + 1
      }
      delta_LogPo <- sum_LogPo - LogPo
      LogPo <- sum_LogPo
    
      message(paste0(
        "Log-Posterior Probability: ", round(LogPo, 2),
        ", ",
        "\u0394: ", .sign_numbers(delta_LogPo), round(delta_LogPo, 2)
      ))
    
      iter <- iter + 1
      if (iter >= limit || stuck > 1) {break}
    }
    multiRL.models[[i]] <- multiRL.model.MAP
  }
  # 停止并行
  future::plan(future::sequential)
  
################################### [ END ] ####################################
  
  # 整理结果成表格
  result <- .extract_results(multiRL.models)
  
  return(result)
}
