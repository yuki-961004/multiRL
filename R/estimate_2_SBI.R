#' @title 
#' Simulated-Based Inference (SBI)
#' @name estimate_2_SBI
#' @description 
#'  Since both Approximate Bayesian Computation (ABC) and Recurrent Neural 
#'    Network (RNN) are simulation-based inference methods, they require a 
#'    model built from a large amount of simulated data before running. This 
#'    model is then used to predict the most likely input parameters 
#'    corresponding to the real data. Therefore, this function generates random 
#'    input parameters using user-specified distributions and produces 
#'    simulated data based on these parameters.
#'
#' @param model 
#'  Reinforcement Learning Model
#' @param env 
#'  multiRL.env
#' @param priors 
#'  Prior probability density function of the free parameters,
#'    see \link[multiRL]{priors}
#' @param control 
#'  Settings manage various aspects of the iterative process,
#'    see \link[multiRL]{control}
#' @param ... 
#'  Additional arguments passed to internal functions.
#'
#' @return A \code{List} containing, for each model, simulated data generated
#'   using randomly sampled parameters.
#' 
estimate_2_SBI <- function(
    model,
    env,
    priors,
    control = list(),
    ...
){
################################ [default] #####################################
  
  # 编译对象函数
  model <- compiler::cmpfun(model)
  
  multiRL.env <- env
  environment(model) <- multiRL.env
  
  default = list(
    # General
    seed = 123,
    core = 1,
    sample = 100,
    # SBI
    train = 1000
    #scope = "individual"
  )
  control <- utils::modifyList(x = default, val = control, keep.null = TRUE)
  
  list2env(control, envir = environment())
  
################################# [params] #####################################
  
  list_params <- list()
  
  for (i in 1:sample) {
    params <- c()
    
    for (j in 1:length(priors)) {
      set.seed(seed + length(priors) * i + j) 
      params[j] <- priors[[j]]()
    }
    
    list_params[[i]] <- params
  }
  
############################### [ register ] ###################################
  
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
  
############################### [ Parallel ] ###################################
  
  message(paste0("Simulating...", "\n"))
  
  list_simulated <- list()
  
  i <- NA
  
  doRNG::registerDoRNG(seed = seed)
  
  suppressMessages({
    list_simulated <- foreach::foreach(
      i = 1:sample, .packages = "multiRL"
    ) %dorng% {
      model(params = list_params[[i]])
    }
  })

############################## [ unregister ] ##################################
  
  future::plan(future::sequential)
  
  return(list_simulated)
}
