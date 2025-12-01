.get_model_info <- function(model) {
  
  # ----- 1. 取得函数名称 -----
  model_name <- .get_function_name(func = model)
  
  # ----- 2. 从函数体内找自由参数的名字 -----
  body_expr <- body(model)
  
  # 找 params <- list(free = list(...)) 这一段
  free_params <- NULL
  
  find_free <- function(expr) {
    if (is.call(expr) && identical(expr[[1]], as.symbol("list"))) {
      # 找 free = list(...)
      args <- as.list(expr)[-1]
      if ("free" %in% names(args) && is.call(args$free)) {
        free_list <- as.list(args$free)[-1]
        return(names(free_list))
      }
    }
    # 递归搜索
    if (is.recursive(expr)) {
      for (e in as.list(expr)) {
        res <- find_free(e)
        if (!is.null(res)) return(res)
      }
    }
    return(NULL)
  }
  
  param_name <- find_free(body_expr)
  
  list(
    model_name = model_name,
    param_name = param_name
  )
}
