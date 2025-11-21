.check_priors_params = function(priors, params) {
  
  # 1. 检查 priors 是否为空 (NULL)
  if (length(priors) == 0) {
    return(invisible(FALSE))
  }
  
  # nocov start
  # 2. 检查名称是否相同
  # 使用 base::all.equal() 检查名称向量是否相同，返回 TRUE 或描述差异的字符串
  name_check <- all.equal(names(priors), names(params))
  
  if (name_check != TRUE) {
    # 名称不匹配：输出信息并返回 FALSE
    message(
      "The names of 'priors' must be identical to the names of 'params'. ",
      "Mismatched names found: \n",
      "  Priors names: ", paste(names(priors), collapse = ", "), "\n",
      "  Params names: ", paste(names(params), collapse = ", ")
    )
    return(invisible(FALSE))
  }
  # nocov end
  
  # 3. 名称匹配：返回 TRUE
  return(invisible(TRUE))
}