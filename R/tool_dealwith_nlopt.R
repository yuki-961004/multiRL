.dealwith_nlopt <- function(algorithm) {
  
  # 因为nlopt会输出两个字符串, 第一个表示全局搜索算法, 第二个表示局部搜素算法
  if (length(algorithm) > 1) {
    global_opts <- algorithm[[1]]
    local_opts <- list(algorithm = algorithm[[2]], xtol_rel = 1e-8)
    algorithm <- "NLOPT"
  } else if (length(algorithm) == 1) {
    global_opts <- algorithm
    local_opts <- list(algorithm = "NLOPT_LN_BOBYQA", xtol_rel = 1e-8)
    algorithm <- "NLOPT"
  } 
  
  result <- list(
    algorithm = algorithm, 
    global_opts = global_opts, 
    local_opts = local_opts
  )
}
