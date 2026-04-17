#' multiRL.behrule
#'
#' @param behrule 
#'  The agent’s implicitly formed internal rule,
#'    see \link[multiRL]{behrule}
#' @param ... 
#'  Additional arguments passed to internal functions.
#' 
#' @return An S4 object of class \code{multiRL.behrule}.
#' 
#'   \describe{
#'     \item{\code{cue}}{
#'       A \code{CharacterVector} containing the cue (state) presented on each 
#'       trial.
#'     }
#'     \item{\code{rsp}}{
#'       A \code{CharacterVector} containing the set of possible actions
#'       available to the agent.
#'     }
#'     \item{\code{extra}}{
#'       A \code{List} containing additional user-defined information.
#'     }
#'   }
#'   
process_2_behrule <- function(
    behrule,
    ...
){
  extra <- list(...)
  
################################## [check] #####################################
  
  # 检查cue, rsp, mid是否都是字符串
  check_type <- all(sapply(behrule, is.character))
  if (!(check_type)) {message("Invalid behrule key type")}
  
  # 默认hidden(mid)
  default <- list(
    mid = c("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
  )
  behrule <- utils::modifyList(x = default, val = behrule)
  
################################ [behrule] #####################################
  
  # behrule -> multiRL.behrule
  multiRL.behrule <- methods::new(
    Class = "multiRL.behrule",
    cue = behrule$cue,
    mid = behrule$mid,
    rsp = behrule$rsp,
    extra = extra
  )
  
  return(multiRL.behrule)
}
