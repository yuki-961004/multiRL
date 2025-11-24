#' estimate_2_RNN
#'
#' @param data data
#' @param behrule behrule
#' @param ids ids
#' @param colnames colnames
#' @param funcs funcs
#' @param priors priors
#' @param settings settings
#' @param models models
#' @param control control
#' @param ... extra
#'
#' @returns params
#' 
estimate_2_RNN <- function(
    data,
    behrule,
    
    ids = NULL,
    colnames,
    funcs = NULL,
    priors,
    settings,
    models,
    
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
  
  # 默认设置
  if (is.null(settings)) {settings <- rep(list(list()), length(models))}
  for (i in 1:length(settings)) {
    default <- list(
      name = paste0("Unknown Model ", i),
      mode = "simulating",
      estimate = "RNN",
      policy = "on"
    )
    settings[[i]] <- utils::modifyList(x = default, val = settings[[i]])
  }
  
  # 默认控制
  default = list(
    seed = 123,
    core = 1,
    
    layer = "GRU",
    info = c(colnames$object, colnames$action),
    units = 128,
    sample = 1000,
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
  # 如果没有输入要拟合的被试序号, 就拟合所有的
  if (is.null(ids)){ids <- dfinfo$all_ids}
  
################################## [RNN] #######################################

  result <- list()
  
  for (i in 1:length(models)) {
    
    if ("name" %in% names(settings[[i]])) {
      model_name <- settings[[i]]$name
    } else {
      model_name <- paste0("Unknown Model ", i)
    }
    
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
        
        result[[i]] <- do.call(rbind, opt_params)
      })
    })
  }
  
  col_order <- c("fit_model", "Subject")
  
  for (i in 1:length(models)) {
    
    result[[i]] <- base::as.data.frame(result[[i]]) 
    base::colnames(result[[i]]) <- base::names(priors[[i]])
    
    result[[i]][["fit_model"]] <- settings[[i]]$name
    result[[i]][["Subject"]] <- ids
    
    result[[i]] <- result[[i]] |>
      dplyr::select(
        dplyr::all_of(col_order), 
        dplyr::everything()
      )
  }
  
  result <- dplyr::bind_rows(result)
  
  return(result)
}
