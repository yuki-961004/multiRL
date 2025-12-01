.get_function_name <- function(func) {
  
  # 1. 先从全局环境找
  objs <- ls(envir = .GlobalEnv)
  for (nm in objs) {
    if (identical(func, get(nm, envir = .GlobalEnv))) {
      return(nm)
    }
  }
  
  # 2. 再从已加载的 package namespace 找
  loaded_pkgs <- loadedNamespaces()
  for (pkg in loaded_pkgs) {
    ns <- getNamespace(pkg)
    objs_ns <- ls(envir = ns)
    for (nm in objs_ns) {
      if (exists(nm, envir = ns)) {
        if (identical(func, get(nm, envir = ns))) {
          return(nm)
        }
      }
    }
  }
  
  # 3. 如果找不到名称 → 这个函数没有名称绑定（匿名）
  return(NULL)
}
