#' @title 
#' Step 1: Building reinforcement learning model
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
#' @param funcs 
#'  The functions forming the reinforcement learning model,
#'    see \link[multiRL]{funcs}
#' @param params 
#'  Parameters used by the model’s internal functions,
#'    see \link[multiRL]{params}
#' @param priors 
#'  Prior probability density function of the free parameters,
#'    see \link[multiRL]{priors}
#' @param settings 
#'  Other model settings, 
#'    see \link[multiRL]{settings}
#' @param engine 
#'  Specifies whether the core Markov Decision Process (MDP) update loop is 
#'    executed in C++ or in R.
#' @param ... 
#'  Additional arguments passed to internal functions.
#'  
#' @return An S4 object of class \code{multiRL.model}.
#'
#'   \describe{
#'     \item{\code{input}}{
#'       An S4 object of class \code{multiRL.input},
#'       containing the raw data, column specifications, parameters and ...
#'     }
#'     \item{\code{behrule}}{
#'       An S4 object of class \code{multiRL.behrule},
#'       defining the latent learning rules.
#'     }
#'     \item{\code{result}}{
#'       An S4 object of class \code{multiRL.result},
#'       storing trial-level outputs of the Markov Decision Process.
#'     }
#'     \item{\code{sumstat}}{
#'       An S4 object of class \code{multiRL.sumstat},
#'       providing summary statistics across different estimation methods.
#'     }
#'     \item{\code{extra}}{
#'       A \code{List} containing additional user-defined information.
#'     }
#'   }
#'   
#' @examples
#' multiRL.model <- multiRL::run_m(
#'   data = multiRL::TAB[multiRL::TAB[, "Subject"] == 1, ],
#'   behrule = list(
#'     cue = c("A", "B", "C", "D"),
#'     rsp = c("A", "B", "C", "D")
#'   ),
#'   colnames = list(
#'     subid = "Subject", block = "Block", trial = "Trial",
#'     object = c("L_choice", "R_choice"), 
#'     reward = c("L_reward", "R_reward"),
#'     action = "Sub_Choose",
#'     exinfo = c("Frame", "NetWorth", "RT")
#'   ),
#'   params = list(
#'     free = list(
#'       alpha = 0.5,
#'       beta = 0.5
#'     ),
#'     fixed = list(
#'       gamma = 1, 
#'       delta = 0.1, 
#'       epsilon = NA_real_,
#'       zeta = 0
#'     ),
#'     constant = list(
#'       seed = 123,
#'       Q0 = NA_real_, 
#'       reset = NA_real_,
#'       lapse = 0.01,
#'       threshold = 1,
#'       bonus = 0,
#'       weight = 1,
#'       capacity = 0,
#'       sticky = 0
#'     )
#'   ),
#'   priors = list(
#'     alpha = function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}, 
#'     beta = function(x) {stats::dexp(x, rate = 1, log = TRUE)}
#'   ),
#'   settings = list(
#'     name = "TD",
#'     mode = "fitting",
#'     estimate = "MLE",
#'     policy = "off",
#'     system = c("RL", "WM")
#'  ),
#'  engine = "R"
#' )
#'  
#' multiRL.summary <- multiRL::summary(multiRL.model)
#' 
#'  
run_m <- function(
    data,
    colnames = list(),
    behrule = list(),
    
    funcs = list(),
    params = list(),
    priors = list(),
    
    settings = list(),
    engine = "Cpp",
    ...
){
  
  multiRL.input <- process_1_input(
    data = data,
    colnames = colnames,
    params = params,
    funcs = funcs,
    priors = priors,
    settings = settings,
    ...
  )
  
  multiRL.behrule <- process_2_behrule(
    behrule = behrule,
    ...
  )
  
  multiRL.record <- process_3_record(
    input = multiRL.input,
    behrule = multiRL.behrule,
    ...
  )
  
  if (engine == "Cpp") {
    multiRL.output <- process_4_output_cpp(
      record = multiRL.record,
      extra = list(...)
    )
  } else if (engine == "R") {
    multiRL.output <- process_4_output_r(
      record = multiRL.record,
      ...
    )
  }
  
  multiRL.metric <- process_5_metric(
    output = multiRL.output,
    ...
  )
  
  multiRL.model <- methods::new(
    Class = "multiRL.model", 
    multiRL.metric
  )
  
  return(multiRL.model)
}
