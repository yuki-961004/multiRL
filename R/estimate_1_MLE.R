#' estimate_1_MLE
#'
#' @param model model
#' @param environment environment
#' @param algorithm algorithm
#' @param lower lower
#' @param upper upper
#' @param initial_params initial_params
#' @param initial_size initial_size
#' @param iteration iteration
#' @param seed seed
#' @param ... extra
#'
#' @returns multiRL.model
#' 
estimate_1_MLE <- function(
    model,
    environment,
    algorithm,
    lower,
    upper,
    initial_params = NA,
    initial_size = 50,
    iteration = 10,
    seed = 123,
    ...
){
  ################################# [before] #####################################
  
  obj_func <- model
  multiRL.env <- environment
  environment(obj_func) <- multiRL.env
  if (length(initial_params) == 1 && is.na(initial_params)){
    initial_params <- lower + 1e-2
  }
  
  ############################### [algorithm] ####################################
  
  if (algorithm[[1]] == "L-BFGS-B") {
    result <- stats::optim(
      method = "L-BFGS-B", fn = obj_func,
      par = initial_params, lower = lower, upper = upper,
      control = list(maxit = iteration)
    )
  } else if (algorithm[[1]] == "GenSA") {
    .check_dependency("GenSA", algorithm_name = "Simulated Annealing")
    
    result <- GenSA::GenSA(
      fn = obj_func,
      par = initial_params, lower = lower, upper = upper,
      control = list(maxit = iteration, seed = seed)
    )
  } else if (algorithm[[1]] == "GA") {
    .check_dependency("GA", algorithm_name = "Genetic Algorithm")
    
    result <- GA::ga(
      type = "real-valued", fitness = function(x) -obj_func(x),
      popSize = initial_size, lower = lower, upper = upper,
      maxiter = iteration, monitor = FALSE
    )
  } else if (algorithm[[1]] == "DEoptim") {
    .check_dependency("DEoptim", algorithm_name = "Differential Evolution")
    
    result <- DEoptim::DEoptim(
      fn = obj_func,
      lower = lower, upper = upper,
      control = DEoptim::DEoptim.control(
        NP = initial_size, itermax = iteration, trace = FALSE
      )
    )
  } else if (algorithm[[1]] == "Bayesian") {
    required_pkgs <- c(
      "mlrMBO", "mlr", "ParamHelpers", "smoof", "lhs",
      "DiceKriging", "rgenoud"
    )
    .check_dependency(required_pkgs, algorithm_name = "Bayesian Optimization")
    
    param_list <- lapply(
      1:length(lower), function(i) {
        ParamHelpers::makeNumericParam(
          id = paste0("param_", i),
          lower = lower[i], upper = upper[i]
        )
      }
    )
    
    bys_func <- smoof::makeSingleObjectiveFunction(
      fn = obj_func, par.set = ParamHelpers::makeParamSet(params = param_list)
    )
    
    suppressWarnings(
      result <- mlrMBO::mbo(
        fun = bys_func, 
        design = ParamHelpers::generateDesign(
          n = initial_size, 
          par.set = ParamHelpers::getParamSet(bys_func), 
          fun = lhs::maximinLHS
        ), 
        control = mlrMBO::setMBOControlInfill(
          mlrMBO::setMBOControlTermination(
            control = mlrMBO::makeMBOControl(),
            iters = iteration
          ),
          opt.focussearch.maxit = 10
        ),
        show.info = FALSE
      )
    )
  } else if (algorithm[[1]] == "PSO") {
    .check_dependency("pso", algorithm_name = "Particle Swarm Optimization")
    
    result <- pso::psoptim(
      par = initial_params, fn = obj_func,
      lower = lower, upper = upper,
      control = list(maxit = iteration, trace = 0)
    )
  } else if (algorithm[[1]] == "CMA-ES") {
    .check_dependency("cmaes", algorithm_name = "Covariance Matrix Adapting")
    
    result <- cmaes::cma_es(
      fn = obj_func,
      par = initial_params, lower = lower, upper = upper,
      control = list(maxit = iteration)
    )
  } else if (startsWith(algorithm[[1]], "NLOPT_")) {
    .check_dependency("nloptr", algorithm_name = "Nonlinear Optimization")
    
    if (length(algorithm) > 1) {
      local_opts <- list(algorithm = algorithm[[2]], xtol_rel = 1.0e-8)
    } else {
      local_opts <- NULL
    }
    
    result <- nloptr::nloptr(
      eval_f = obj_func,
      x0 = initial_params, lb = lower, ub = upper,
      opts = list(
        algorithm = algorithm[[1]], local_opts = local_opts, maxeval = iteration
      )
    )
  } 
  
  ############################# [optimal params] #################################
  
  if (algorithm[[1]] == "L-BFGS-B") {
    fit_params <- as.vector(result$par)
  } else if (algorithm[[1]] == "GenSA") {
    fit_params <- as.vector(result$par)
  } else if (algorithm[[1]] == "GA") {
    fit_params <- as.vector(result@solution[1,])
  } else if (algorithm[[1]] == "DEoptim") {
    fit_params <- as.vector(result$optim$bestmem)
  } else if (algorithm[[1]] == "Bayesian") {
    fit_params <- as.vector(
      as.numeric(result$final.opt.state$opt.result$mbo.result$x)
    )
  } else if (algorithm[[1]] == "PSO") {
    fit_params <- as.vector(result$par)
  } else if (algorithm[[1]] == "CMA-ES") {
    fit_params <- as.vector(result$par)
  } else if (startsWith(algorithm[[1]], "NLOPT_")) {
    fit_params <- as.vector(result$solution)
  } 
  
  obj_func(params = fit_params)
  
  return(multiRL.env$multiRL.model)
}