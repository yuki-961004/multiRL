#' behrule
#'
#' @param behrule behrule
#' @param ... extra
#'
#' @returns multiRL.behrule
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
