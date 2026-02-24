#' @title 
#' Estimation Method: Approximate Bayesian Computation (ABC)
#' @name estimate_2_ABC
#' @description 
#'  This function takes a large set of simulated data to train an Approximate 
#'    Bayesian Computation (ABC) model and then uses the trained model to 
#'    estimate optimal parameters for the target data.
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
#' @returns An S3 object of class \code{DataFrame} containing, for each model, 
#'  the estimated optimal parameters and associated model fit metrics.
#' 
estimate_2_ABC <- function(
    data, 
    colnames,
    behrule,
    ids = NULL,
    
    models,
    funcs = NULL,
    priors,
    settings = NULL,
    
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
  
  # 默认设置
  settings <- .restructure_settings(x = settings, n = length(models))
  for (i in 1:length(settings)) {
    default <- list(
      name = paste0("Unknown_", i),
      policy = "on"
    )
    settings[[i]] <- utils::modifyList(x = default, val = settings[[i]])
  }
  
  # 强制设置
  for (i in 1:length(settings)) {
    settings[[i]]$mode <- "fitting"
    settings[[i]]$estimate <- "ABC"
    settings[[i]]$policy <- "on"
  }
  
  # 转换先验
  priors <- .convert_priors(priors = priors, to = "rfunc")
  
  # 默认控制
  default = list(
    # simulate
    seed = 123,
    core = 1,
    # abc
    sample = 100,
    train = 1000,
    scope = "individual",
    tol = 0.1,
    reduction = "NONE"
  )
  control <- utils::modifyList(x = default, val = control)
  # 解放control中的设定, 变成全局变量
  list2env(control, envir = environment())
  
############################ [aotu-detect data] ################################
  
  # 自动探测数据
  suppressMessages({dfinfo <- .detect_data(data)})
  # 如果没有输入被试序号的列名. 则自动探测
  if ("subid" %in% names(colnames)) {
    subid <- colnames[["subid"]]
  } else {
    subid <- dfinfo$sub_col_name
  }
  # 如果没有输入要拟合的被试序号, 就拟合所有的被试
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
  result.ABC <- list()
  # 定义foreach内的指针
  j <- NA
  
  for (i in 1:length(models)) {
    
    model_name <- settings[[i]]$name
    n_params <- length(priors[[i]])

################################## [ABC] #######################################
    
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
      # ABC启动!
      suppressMessages({
        opt_params <- list()
        
        if ( scope == "shared" ) {
          
          ABC <- engine_ABC(
            data = data[data[, subid] == 1, ],
            behrule = behrule,
            colnames = colnames,
            funcs = funcs[[i]],
            settings = settings[[i]],
            priors = priors[[i]],
            model = models[[i]],
            control = control
          )
          
          opt_params <- foreach::foreach(
            j = ids, .packages = c("multiRL")
          ) %dorng% {
            sub_data <- data[data[, subid] == j, ]
            
            target <- .for_abc(
              data = sub_data, rsp = behrule$rsp,
              block = colnames$block, action = colnames$action
            )
            
            reduced_sumstats <- .reduce_sumstats(
              method = reduction, abc = ABC, target = target
            ) 
            
            utils::capture.output(
              suppressWarnings({
                opt_params_j <- summary(abc::abc(
                  target = reduced_sumstats$target$onerow, 
                  param = reduced_sumstats$abc$df_params, 
                  sumstat = reduced_sumstats$abc$df_sumstats, 
                  tol = tol, 
                  method = "neuralnet", 
                  transf = rep("logit", n_params),
                  logit.bounds = cbind(lowers[[i]], uppers[[i]])
                ))[5, ]
              })
            )
            
            names(opt_params_j) <- names(priors)
            p()
            return(opt_params_j)
          }       
        } else if ( scope == "individual" ) {

          opt_params <- foreach::foreach(
            j = ids, .packages = c("multiRL")
          ) %dorng% {
            sub_data <- data[data[, subid] == j, ]
            
            ABC <- engine_ABC(
              data = sub_data,
              behrule = behrule,
              colnames = colnames,
              funcs = funcs[[i]],
              settings = settings[[i]],
              priors = priors[[i]],
              model = models[[i]],
              control = control
            )
            
            target <- .for_abc(
              data = sub_data, rsp = behrule$rsp,
              block = colnames$block, action = colnames$action
            )
            
            reduced_sumstats <- .reduce_sumstats(
              method = reduction, abc = ABC, target = target
            ) 
            
            utils::capture.output(
              suppressWarnings({
                opt_params_j <- summary(abc::abc(
                  target = reduced_sumstats$target$onerow, 
                  param = reduced_sumstats$abc$df_params, 
                  sumstat = reduced_sumstats$abc$df_sumstats, 
                  tol = tol, 
                  method = "neuralnet", 
                  transf = rep("logit", n_params),
                  logit.bounds = cbind(lowers[[i]], uppers[[i]])
                ))[5, ]
              })
            )
            
            names(opt_params_j) <- names(priors)
            p()
            return(opt_params_j)
          }
        }
        result.ABC[[i]] <- do.call(rbind, opt_params)
      })
    })
  }
  
  # 停止并行
  future::plan(future::sequential)

  col_order <- c("fit_model", "Subject")
  
  for (i in 1:length(models)) {
    
    result.ABC[[i]] <- as.data.frame(result.ABC[[i]]) 
    colnames(result.ABC[[i]]) <- names(priors[[i]])
    # 新增两列作为序号
    result.ABC[[i]][["fit_model"]] <- settings[[i]]$name
    result.ABC[[i]][["Subject"]] <- ids
    # 找到原始列的名字
    remaining_cols <- setdiff(names(result.ABC[[i]]), col_order)
    # 序号列 + 数据列
    result.ABC[[i]] <- result.ABC[[i]][c(col_order, remaining_cols)]
    
  }
  
  result.ABC <- .rbind_fill(result.ABC)
  
  return(result.ABC)  
}