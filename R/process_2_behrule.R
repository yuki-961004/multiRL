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
  
  key_names <- c("cue", "rsp")
  # 检查colnames的元素名称是否一致
  check_key <- all(sort(names(behrule)) == sort(key_names))
  if (!(check_key)) {message("Invalid behrule keys")}
  # 检查colnames的元素类型是否一致
  check_type <- all(sapply(behrule, is.character))
  if (!(check_type)) {message("Invalid behrule key type")}
  
################################ [behrule] #####################################
  
  # behrule -> multiRL.behrule
  multiRL.behrule <- methods::new(
    Class = "multiRL.behrule",
    cue = behrule$cue,
    rsp = behrule$rsp,
    extra = extra
  )
  
  return(multiRL.behrule)
}
