#' multiRL.record
#'
#' @param input 
#'  multiRL.input
#' @param behrule 
#'  multiRL.behrule
#' @param ... 
#'  Additional arguments passed to internal functions.
#'  
#' @return An S4 object of class \code{multiRL.record}.
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
#'       An S4 object of class \code{multiRL.result}, which is empty for now, 
#'       storing trial-level outputs of the Markov Decision Process.
#'     }
#'     \item{\code{extra}}{
#'       A \code{List} containing additional user-defined information.
#'     }
#'   }
#' 
process_3_record <- function(
    input,
    behrule,
    ...
){
  
################################## [check] #####################################
  
  extra <- list(...)
  
  system <- input@settings@system
  n_system <- length(system)
  
############################### [null matrix] ##################################
  
  # 生成行数等量的多列表格
  nulldf <- matrix(
    data = NA_real_,
    nrow = input@n_rows,
    ncol = length(behrule@cue)
  )
  # 列数表示需要更新的价值, 即潜在规则
  colnames(nulldf) <- behrule@cue
  
  # 生成行数等量的单列表格
  singledf <- matrix(
    data = NA_real_,
    nrow = input@n_rows,
    ncol = 1
  )
  
  value <- replicate(n = n_system, expr = nulldf, simplify = FALSE)
  names(value) <- system
  
  bias        <- nulldf
  shown       <- nulldf
  prob        <- nulldf
  count       <- nulldf
  
  exploration <- matrix(as.numeric(singledf), nrow = nrow(singledf), ncol = 1)
  latent      <- matrix(as.character(singledf), nrow = nrow(singledf), ncol = 1)
  reward      <- matrix(as.numeric(singledf), nrow = nrow(singledf), ncol = 1)
  utility     <- matrix(as.numeric(singledf), nrow = nrow(singledf), ncol = 1)
  simulation  <- matrix(as.character(singledf), nrow = nrow(singledf), ncol = 1)
  position    <- matrix(as.character(singledf), nrow = nrow(singledf), ncol = 1)
  
  result <- methods::new(
    Class = "multiRL.result",
    value = value,
    bias = bias,
    shown = shown,
    prob = prob,
    count = count,
    
    exploration = exploration,
    latent = latent,
    reward = reward,
    utility = utility,
    simulation = simulation,
    position = position
  )
  
################################## [record] ####################################

  multiRL.record <- methods::new(
    Class = "multiRL.record",
    input = input,
    behrule = behrule,
    result = result,
    extra = extra
  )
  
  return(multiRL.record)
}
