.check_priors_params = function(priors, params) {
  
  # 1. 如果没有输入priors返回FALSE
  if (length(priors) == 0) {
    return(invisible(FALSE))
  }
  
  # nocov start
  # 2. 检查priors和params的名称是否相同
  name_check <- all.equal(names(priors), names(params))
  
  if (name_check != TRUE) {
    # 如果有的参数有先验有的没有, 则返回FALSE
    message(
      "The names of 'priors' must be identical to the names of 'params'. ",
      "Mismatched names found: \n",
      "  Priors names: ", paste(names(priors), collapse = ", "), "\n",
      "  Params names: ", paste(names(params), collapse = ", ")
    )
    return(invisible(FALSE))
  }
  # nocov end
  
  # 3. 前面的情况都没出现, 才返回TRUE
  return(invisible(TRUE))
}