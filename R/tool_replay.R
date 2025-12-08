.replay <- function(
    data, 
    behrule,
    
    ids = NULL,
    colnames,
    funcs = NULL,
    priors = NULL,
    settings = NULL,
    
    result,
    models,
    free_params = NULL,
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
  if (is.null(priors)) {priors <- rep(list(list()), length(models))}
  
  # 默认设置
  if (is.null(settings)) {settings <- rep(list(list()), length(models))}
  for (i in 1:length(settings)) {
    default <- list(
      name = paste0("Unknown_", i),
      mode = "replay",
      policy = "on"
    )
    settings[[i]] <- utils::modifyList(x = default, val = settings[[i]])
  }
  
############################ [aotu-detect data] ################################
  
  # 获取被试id所在列
  suppressMessages({dfinfo <- .detect_data(data)})
  subid <- dfinfo$sub_col_name
  if (is.null(ids)){ids = unique(result$Subject)}
  
  model_info <- list()
  if (is.null(free_params)) {
    for (i in 1:length(models)) {
      model_info[[i]] <- .get_model_info(model = models[[i]])
    }
  }
  
  # 检查 fit_model
  if ("fit_model" %in% colnames(result)) {
    col_model <- "fit_model"
  }
  
  # 检查 simulate_model
  if ("simulate_model" %in% colnames(result)) {
    col_model <- "simulate_model"
  }
  
################################# [replay] #####################################
  
  multiRL.models <- list()
  
  for (i in 1:length(models)) {
    model_name <- model_info[[i]]$model_name
    param_name <- model_info[[i]]$param_name
    
    model_params <- result[
      result[[col_model]] == model_name, c(subid, param_name)
    ]
    
    list_params <- split(
      x = model_params[setdiff(names(model_params), subid)],
      f = model_params[[subid]]
    )
    
    multiRL.models[[i]] <- list()
    names(multiRL.models)[i] <- model_name
    
    for (j in ids) {
      env <- estimate_0_ENV(
        data = data[data[, subid] == j, ],
        behrule = behrule,
        colnames = colnames,
        funcs = funcs[[i]],
        priors = priors[[i]],
        settings = settings[[i]]
      )
      
      multiRL.env <- env
      environment(models[[i]]) <- multiRL.env
      
      multiRL.models[[i]][[j]] <- models[[i]](
        params = as.numeric(list_params[[as.character(j)]])
      )
    }
  }
  
  return(multiRL.models)
}
