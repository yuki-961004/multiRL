.get_dfunc <- function(func) {
  
  # 1. 获取函数体
  func_body <- body(func)
  
  # 2. 检查并提取核心调用表达式
  # 如果函数体是 {...} 块 (class(func_body) == "call" 且 func_body[[1]] == "{")
  if (is.call(func_body) && as.character(func_body[[1]]) == "{") {
    
    # 提取代码块中的最后一个表达式 (即代码块的最后一个元素)
    # 这符合R函数通常返回最后一个表达式值的习惯
    core_expr <- func_body[[length(func_body)]]
    
  } else if (is.call(func_body)) {
    
    # 如果函数体是直接的调用（没有被 {} 包裹，不常见，但合法）
    core_expr <- func_body
    
  } else {
    # 既不是 Call 也不是 {} 块 (如 NULL 或非表达式)
    stop("Error: Function body is not a valid expression or call.")
  }
  
  # --- 从这里开始，分析 core_expr ---
  
  # 3. 提取调用对象（Call object）的函数名部分
  func_call_symbol <- core_expr[[1]]
  
  # 4. 提取函数名 Symbol/Name
  
  # 如果函数名是带命名空间的 (:: 或 :::) 的 Call，例如 stats::dbeta
  if (is.call(func_call_symbol)) {
    
    # 提取实际的函数名（在 Call 的第三个位置）
    dist_name <- as.character(func_call_symbol[[3]])
    
    # 额外检查命名空间是否为 stats (仅作提示，不影响结果)
    namespace <- as.character(func_call_symbol[[2]])
    if (namespace != "stats" && namespace != "base") {
      warning(
        "Warning: Function is not called from the 'stats' or 'base' package."
      )
    }
    
  } else {
    # 如果函数名是一个简单的 Symbol，例如 dbeta
    dist_name <- as.character(func_call_symbol)
  }
  
  # 5. 检查并返回结果
  if (startsWith(dist_name, "d") && nchar(dist_name) > 1) {
    
    # 截掉开头的 'd'
    # base::substr
    return(base::substr(dist_name, 2, nchar(dist_name)))
    
  } else {
    
    # 处理你的原始错误输出
    return(paste0(
      "Unknown: Found call '", dist_name, "' which does not start with 'd'"
    ))
  }
}
