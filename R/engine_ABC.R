#' engine_ABC
#'
#' @param data data
#' @param behrule behrule
#' @param colnames colnames
#' 
#' @param model model
#' @param funcs funcs
#' @param priors priors
#' @param settings settings
#' 
#' @param control control
#'
#' @returns df_params and df_sumstats
#' 
engine_ABC <- function(
    data = data,
    behrule = behrule,
    colnames,
    
    model,
    funcs = NULL,
    priors,
    settings = NULL,
    
    control = control
){
  
  # 确保训练模型和参数恢复时没有用同一份数据
  control$seed <- control$seed * 2
  # 训练模型的样本量是train, 检测模型的量是sample
  control$sample <- control$train
  
############################### [Simulate] #####################################
  
  multiRL.env <- estimate_0_ENV(
    data = data,
    behrule = behrule,
    colnames = colnames,
    funcs = funcs,
    settings = settings,
  )
  
  list_simulated <- estimate_2_SBI(
    model = model,
    env = multiRL.env,
    priors = priors,
    control = control
  )
  
############################### [Simulate] #####################################
  
  #Step 1: Free Parameters
  list_params <- lapply(list_simulated, .extract_params)
  df_params <- as.data.frame(do.call(rbind, list_params))
  
  #Step 2: Summary Statistics
  list_sumstats <- lapply(list_simulated, .extract_sumstats)
  df_sumstats <- as.data.frame(do.call(rbind, list_sumstats))
  
  ABC <- list(
    df_params = df_params,
    df_sumstats = df_sumstats
  )
    
  return(ABC)
}