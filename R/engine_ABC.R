#' engine_ABC
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
#' @param model 
#'  Reinforcement Learning Model
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
#' @return A \code{List} containing a \code{DataFrame} of the parameters used to 
#'    generate the simulated data and the summary statistics for Approximate
#'    Bayesian Computation (ABC).
#' 
engine_ABC <- function(
    data,
    colnames,
    behrule,
    
    model,
    funcs = NULL,
    priors,
    
    settings = NULL,
    control = control,
    ...
){
  
  # 确保训练模型和参数恢复时没有用同一份数据
  control$seed <- control$seed * 2
  # 训练模型的样本量是train, 检测模型的量是sample
  control$sample <- control$train
  # 因为abc的部分需要多核, 所以这里必须是单核
  control$core <- 1
  
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