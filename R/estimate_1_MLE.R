#' estimate_1_MLE
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
estimate_1_MLE <- function(
    data, 
    behrule,
    
    ids = NULL,
    colnames,
    funcs = NULL,
    priors,
    settings,
    models,
    
    algorithm,
    lowers,
    uppers,
    control,
    ...
){

################################ [default] #####################################
  
  default <- list(
    subid = "Subject", 
    block = "Block", 
    trial = "Trial",
    object = NA_character_, 
    reward = NA_character_, 
    action = "Action"
  )
  colnames <- utils::modifyList(x = default, val = colnames)
  
  if (is.null(funcs)) {funcs <- rep(list(list()), length(models))}
  for (i in 1:length(funcs)) {
    default <- list(
      rate_func = multiRL::func_alpha,
      prob_func = multiRL::func_beta,
      util_func = multiRL::func_gamma,
      bias_func = multiRL::func_delta,
      expl_func = multiRL::func_epsilon
    )
    funcs[[i]] <- utils::modifyList(x = default, val = funcs[[i]])
  }

  default = list(
    pars = NA,
    size = 50,
    iter = 10,
    seed = 123,
    core = 1
  )
  control <- utils::modifyList(x = default, val = control)
  
  list2env(control, envir = environment())
  
############################ [aotu-detect data] ################################
  
  # 自动探测数据
  suppressMessages({dfinfo <- .detect_data(data)})
  # 如果没有输入被试序号的列名. 则自动探测
  if ("subid" %in% names(colnames)) {subid <- colnames[["subid"]]} 
  else {subid <- dfinfo$sub_col_name}
  
  # 如果没有输入要拟合的被试序号, 就拟合所有的
  if (is.null(ids)){ids <- dfinfo$all_ids}
  
############################### [ Parallel ] ###################################  
  
  sys <- Sys.info()[["sysname"]]
  
  if (core == 1) {
    future::plan(future::sequential)
  } 
  # Windows
  else if (sys == "Windows") {
    future::plan(future::multisession, workers = core)
  } 
  # macOS
  else if (sys == "Darwin") {  
    future::plan(future::multisession, workers = core)
  } 
  # Linux
  else if (sys == "Linux") {
    future::plan(future::multicore, workers = core)
  }
  
  doFuture::registerDoFuture()
  
################################### [ MLE ] #################################### 
  
  # 创建空list, 用于存放结果
  multiRL.models <- list(list(), list(), list())
  j <- NA
  
  for (i in 1:length(models)) {
    
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
        
        estimate_1_LBI(
          model = models[[i]],
          env = env,
          algorithm = algorithm,
          lower = lowers[[i]],
          upper = uppers[[i]],
          control = control
        )
      }
    })
  }
  
  future::plan(future::sequential)
  
  result <- .extract_results(
    multiRL.models, 
    n_model = length(models), n_subject = length(ids)
  )
}
