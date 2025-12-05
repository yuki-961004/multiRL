.check_priors_params <- function(priors, params) {
  
  # 1. 快速短路: 长度为0直接失败
  if (length(priors) == 0L) return(FALSE)
  
  # 2. 核心比对: 使用 identical 替代 all.equal
  # names() 是 base 函数, 访问属性极快
  # 只有当 names 完全一致(包括顺序)时, identical 才返回 TRUE
  if (identical(names(priors), names(params))) {
    return(TRUE)
  }
  
  # 3. 错误处理 (仅在失败时执行, 节省 Happy Path 开销)
  # nocov start
  message(
    "The names of 'priors' must be identical to the names of 'params'. ",
    "Mismatched names found: \n",
    "  Priors names: ", paste(names(priors), collapse = ", "), "\n",
    "  Params names: ", paste(names(params), collapse = ", ")
  )
  return(FALSE)
  # nocov end
}