process_2_rule <- function(
  rule,
  ...
){
  ################################## [check] #####################################
  # 额外信息
  extra <- list(...)
  
  key_names <- c("cue", "rsp")
  # 检查colnames的元素名称是否一致
  check_key <- all(sort(names(rule)) == sort(key_names))
  if (!(check_key)) {message("Invalid rule keys")}
  # 检查colnames的元素类型是否一致
  check_type <- all(sapply(rule, is.character))
  if (!(check_type)) {message("Invalid rule key type")}
  
  # rule -> multiRL.behrules
  methods::setClass(
    Class = "multiRL.behrules",
    slots = list(
      cue = "character", 
      rsp = "character",
      extra = "list"
    )
  )
  
  multiRL.behrules <- methods::new(
    Class = "multiRL.behrules",
    cue = rule$cue,
    rsp = rule$rsp,
    extra = extra
  )
  
  return(multiRL.behrules)
}
