#' estimate_2_SBI
#'
#' @param model model
#' @param env multiRL.env
#' @param priors priors
#' @param control control
#' @param ... extra
#'
#' @returns simulated data
#' 
estimate_2_SBI <- function(
    model,
    env,
    priors,
    control = list(),
    ...
){
################################ [default] #####################################
  
  multiRL.env <- env
  environment(model) <- multiRL.env
  
  default = list(
    core = 1,
    iter = 10,
    seed = 123
  )
  control <- utils::modifyList(x = default, val = control)
  
  list2env(control, envir = environment())
  
################################# [params] #####################################
  
  list_params <- list()
  
  for (i in 1:iter) {
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
  
  progressr::handlers(progressr::handler_txtprogressbar)
  
  progressr::with_progress({
    
    p <- progressr::progressor(steps = iter)
    
    i <- NA
    
    doRNG::registerDoRNG(seed = seed)
    
    suppressMessages({
      list_simulated <- foreach::foreach(
        i = 1:iter, .packages = "multiRL"
      ) %dorng% {
        model(params = list_params[[i]])
        p()
      }
    })
    
  })

############################## [ unregister ] ##################################
  
  future::plan(future::sequential)
  
  return(list_simulated)
}
