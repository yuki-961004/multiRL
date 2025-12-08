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
      dcay_func = multiRL::func_zeta
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
      policy = "off"
    )
    settings[[i]] <- utils::modifyList(x = default, val = settings[[i]])
  }
  
  # 强制设置
  for (i in 1:length(settings)) {
    settings[[i]]$mode <- "fitting"
    settings[[i]]$estimate <- "MAP"
  }
  
  # 默认控制
  default = list(
    pars = NA,
    size = 50,
    iter = c(10, 10),
    seed = 123,
    core = 1,
    diff = 0.001,
    patience = 10
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
    
    multiRL.model.MAP.best <- multiRL.model.MLE
    
######################### [ Initialize Posteriors ] ############################
    
    posteriors <- .update_priors(x = multiRL.model.MLE, priors = priors[[i]])
    sum_LogPo <- sum(sapply(multiRL.model.MLE, function(x) x@sumstat@LPo))
    delta_LogPo = 1
    LogPo <- sum_LogPo
    
    if (is.infinite(LogPo)) {
      LogPo <- 0
      warning(paste0(
        "Infinite log-priors detected. Please adjust the priors."
      ))
    }
    message(paste0(
      "Starting Expectation-Maximization Algorithm", "\n",
      "Log-Posterior Probability: ", round(LogPo, 2)
    ))
    
################################### [ MAP ] #################################### 
    
    iter <- 0
    stuck <- 0
    
    hp <- patience
    best_LogPo <- -Inf
    
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
      # 如果出现了Inf, 则说明先验被调整出了问题
      if (is.infinite(LogPo) || is.infinite(sum_LogPo)) {
        LogPo <- 0
        sum_LogPo <- 0
        warning(paste0(
          "Infinite log-priors detected. Please adjust the priors."
        ))
      } else if (delta_LogPo == LogPo - sum_LogPo) {
        stuck <- stuck + 1
      }
      delta_LogPo <- sum_LogPo - LogPo
      LogPo <- sum_LogPo
    
      # 如果这次没有改进, 则耐心-1, 如果耐心为0, 则会提前结束
      if (LogPo > best_LogPo) {
        best_LogPo <- LogPo
        hp <- hp + 1
        multiRL.model.MAP.best <- multiRL.model.MAP
      } else {
        hp <- hp - 1
      }
      
      message(paste0(
        "current: ", round(LogPo, 2),
        ", ",
        "\u0394: ", .sign_numbers(delta_LogPo), round(delta_LogPo, 3),
        ", ",
        "best: ", round(best_LogPo, 2), 
        ", ", 
        "patience: ", hp
      ))
    
      iter <- iter + 1
      
      if (abs(delta_LogPo) <= diff ) {
        message(paste0(
          "Congrets~ EM-MAP finds solution!"
        ))
      } else if (iter >= limit) {
        message(paste0(
          "Iteration limit reached without convergence."
        ))
        break
      } else if (stuck > 1 || hp < 0) {
        message(paste0(
          "EM-MAP seems to be stuck",
          ". ",
          "You could try other priors or just accept the best results for now."
        ))
        break
      }
    }
    multiRL.models[[i]] <- multiRL.model.MAP.best
  }
  # 停止并行
  future::plan(future::sequential)
  
################################### [ END ] ####################################
  
  # 整理结果成表格
  result <- .extract_results(multiRL.models)
  
  return(result)
}
