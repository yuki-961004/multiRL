#' estimate_1_LBI
#'
#' @param model 
#'  Reinforcement Learning Model
#' @param env 
#'  multiRL.env
#' @param algorithm 
#'  Algorithm packages that multiRL supports, 
#'    see \link[multiRL]{algorithm}
#' @param lower 
#'  Lower bound of free parameters
#' @param upper 
#'  Upper bound of free parameters
#' @param control 
#'  Settings manage various aspects of the iterative process,
#'    see \link[multiRL]{control}
#' @param ... 
#'  Additional arguments passed to internal functions.
#'
#' @returns An S4 object of class \code{multiRL.model} 
#'  generated using the estimated optimal parameters.
#' 
estimate_1_LBI <- function(
    model,
    env,
    algorithm,
    lower,
    upper,
    control = list(),
    ...
){
  
  # 编译对象函数
  model <- compiler::cmpfun(model)
  # 所以对象函数所处环境
  multiRL.env <- env
  environment(model) <- multiRL.env
  
################################ [default] #####################################
  
  # 默认控制
  default = list(
    pars = NA,
    dash = 1e-5,
    size = 50,
    iter = 10,
    seed = 123
  )
  control <- utils::modifyList(x = default, val = control)
  
  list2env(control, envir = environment())
  
  # 如果采用了MAP, 则以第一个值为MLE的迭代次数
  if (length(iter) > 1) {iter <- iter[1]} 
  # 迭代的初始值
  if (length(pars) == 1 && is.na(pars)){pars <- lower + 1e-2}
  
  lower <- lower + dash
  upper <- upper - dash
  
################################# [nloptr] #####################################
  
  if (startsWith(algorithm[[1]], "NLOPT_")) {
    opts <- .dealwith_nlopt(algorithm = algorithm)
    algorithm <- opts$algorithm
    global_opts <- opts$global_opts
    local_opts <- opts$local_opts
  }
  
  set.seed(seed)
  
############################### [algorithm] ####################################
  
  result <- switch(
    EXPR = algorithm,
    "L-BFGS-B" = {
      stats::optim(
        method = "L-BFGS-B", fn = model,
        par = pars, lower = lower, upper = upper,
        control = list(maxit = iter)
      )
    },
    "GenSA" = {
      .check_dependency("GenSA", algorithm_name = "Simulated Annealing")
      
      GenSA::GenSA(
        fn = model,
        par = pars, lower = lower, upper = upper,
        control = list(maxit = iter, seed = seed)
      )
    },
    "GA" = {
      .check_dependency("GA", algorithm_name = "Genetic Algorithm")
      
      GA::ga(
        type = "real-valued", fitness = function(x) -model(x),
        popSize = size, lower = lower, upper = upper,
        maxiter = iter, monitor = FALSE, seed = seed
      )
    },
    "DEoptim" = {
      .check_dependency("DEoptim", algorithm_name = "Differential Evolution")
      
      DEoptim::DEoptim(
        fn = model,
        lower = lower, upper = upper,
        control = DEoptim::DEoptim.control(
          NP = size, itermax = iter, trace = FALSE
        )
      )
    },
    "Bayesian" = {
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
        fn = model, par.set = ParamHelpers::makeParamSet(params = param_list)
      )
      
      suppressWarnings(
        mlrMBO::mbo(
          fun = bys_func, 
          design = ParamHelpers::generateDesign(
            n = size, 
            par.set = ParamHelpers::getParamSet(bys_func), 
            fun = lhs::maximinLHS
          ), 
          control = mlrMBO::setMBOControlInfill(
            mlrMBO::setMBOControlTermination(
              control = mlrMBO::makeMBOControl(),
              iters = iter
            ),
            opt.focussearch.maxit = 10
          ),
          show.info = FALSE
        )
      )
    },
    "PSO" = {
      .check_dependency("pso", algorithm_name = "Particle Swarm Optimization")
      
      pso::psoptim(
        par = pars, fn = model,
        lower = lower, upper = upper,
        control = list(maxit = iter, trace = 0)
      )
    },
    "CMA-ES" = {
      .check_dependency("cmaes", algorithm_name = "Covariance Matrix Adapting")
      
      cmaes::cma_es(
        fn = model,
        par = pars, lower = lower, upper = upper,
        control = list(maxit = iter)
      )
    },
    "NLOPT" = {
      .check_dependency("nloptr", algorithm_name = "Nonlinear Optimization")
      
      nloptr::nloptr(
        eval_f = model,
        x0 = pars, lb = lower, ub = upper,
        opts = list(
          algorithm = global_opts, local_opts = local_opts, 
          maxeval = iter, ranseed = seed
        )
      )
    },
  )
  
############################# [optimal params] #################################
  
  opt_params <- switch(
    EXPR = algorithm,
    "L-BFGS-B" = {as.vector(result$par)},
    "GenSA" = {as.vector(result$par)},
    "GA" = {as.vector(result@solution[1,])},
    "DEoptim" = {as.vector(result$optim$bestmem)},
    "Bayesian" = {
      as.vector(as.numeric(result$final.opt.state$opt.result$mbo.result$x))
    },
    "PSO" = {as.vector(result$par)},
    "CMA-ES" = {as.vector(result$par)},
    "NLOPT" = {as.vector(result$solution)},
  )
  
  model(params = opt_params)
  
  return(multiRL.env$multiRL.model)
}
