#' estimate_2_RNN
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
#' @param control 
#'  Settings manage various aspects of the iterative process,
#'    see \link[multiRL]{control}
#' @param ... 
#'  Additional arguments passed to internal functions.
#'
#' @returns data.frame
#' 
estimate_2_RNN <- function(
    data,
    colnames,
    behrule,
    ids = NULL,
    
    models,
    funcs = NULL,
    priors,
    settings,

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
  if (is.null(settings)) {settings <- rep(list(list()), length(models))}
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
    settings[[i]]$estimate <- "RNN"
  }
  
  # 转换先验
  priors <- .convert_priors(priors = priors, to = "rfunc")
  
  # 默认控制
  default = list(
    # simulate
    seed = 123,
    core = 1,
    # tensorflow
    sample = 100,
    train = 1000,
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
  
################################## [RNN] #######################################

  result.RNN <- list()
  
  for (i in 1:length(models)) {
    
    model_name <- settings[[i]]$name
    
    message(paste0(
      "Fitting ", model_name,"\n"
    ))
    
    # 定义进度条
    progressr::handlers(progressr::handler_txtprogressbar)
    # 进度条启动
    progressr::with_progress({
      # 进度条参照
      p <- progressr::progressor(steps = length(ids))

      # RNN启动!
      suppressMessages({
        opt_params <- list()
        
        if ( scope == "shared" ) {
          
          # 只训练一个RNN
          RNN <- engine_RNN(
            data = data[data[, subid] == 1, ],
            behrule = behrule,
            colnames = colnames,
            funcs = funcs[[i]],
            settings = settings[[i]],
            priors = priors[[i]],
            model = models[[i]],
            control = control
          )
          
          for (j in ids) {
            
            sub_data <- data[data[, subid] == j, ]
            
            n_info   <- length(info)
            n_params <- length(priors[[i]])
            n_trials <- nrow(sub_data)
            
            # 预测真实数据对应的参数
            X_sub <- array(NA, dim = c(1, n_trials, n_info))
            X_sub[1, , ] <- .df2matrix(df = sub_data)[, info, drop = FALSE]
            X_pred <- stats::predict(object = RNN, x = X_sub, verbose = 0)
            names(X_pred) <- names(priors)
            opt_params[[j]] <- X_pred
            p()
          }
          
        } else if ( scope == "individual" ) {
          
          for (j in ids) {
            
            sub_data <- data[data[, subid] == j, ]
            
            n_info   <- length(info)
            n_params <- length(priors[[i]])
            n_trials <- nrow(sub_data)
            
            # 为每个被试单独训练模型
            RNN <- engine_RNN(
              data = sub_data,
              behrule = behrule,
              colnames = colnames,
              funcs = funcs[[i]],
              settings = settings[[i]],
              priors = priors[[i]],
              model = models[[i]],
              control = control
            )
            # 预测真实数据对应的参数
            X_sub <- array(NA, dim = c(1, n_trials, n_info))
            X_sub[1, , ] <- .df2matrix(df = sub_data)[, info, drop = FALSE]
            X_pred <- stats::predict(object = RNN, x = X_sub, verbose = 0)
            names(X_pred) <- names(priors)
            opt_params[[j]] <- X_pred
            p()
          }
        }
        result.RNN[[i]] <- do.call(rbind, opt_params)
      })
    })
  }
  
  col_order <- c("fit_model", "Subject")
  
  for (i in 1:length(models)) {
    
    result.RNN[[i]] <- as.data.frame(result.RNN[[i]]) 
    colnames(result.RNN[[i]]) <- names(priors[[i]])
    # 新增两列作为序号
    result.RNN[[i]][["fit_model"]] <- settings[[i]]$name
    result.RNN[[i]][["Subject"]] <- ids
    # 找到原始列的名字
    remaining_cols <- setdiff(names(result.RNN[[i]]), col_order)
    # 序号列 + 数据列
    result.RNN[[i]] <- result.RNN[[i]][c(col_order, remaining_cols)]
    
  }
  
  result.RNN <- .rbind_fill(result.RNN)
  
  return(result.RNN)
}
