.extract_results <- function(x) {
  
  n_model <- length(x)
  n_subject <- length(x[[1]])
  
  # 必须先定义 .flatten_params 辅助函数
  .flatten_params <- function(free_params) {
    
    # unlist 自动将向量元素命名为 "alpha1", "alpha2"
    param_vector <- unlist(x = free_params, recursive = TRUE)
    
    # tibble::as_tibble_row 避免了创建中间数据框
    return(as.data.frame(as.list(param_vector)))
  }
  
  # 外部循环 (i: model)
  final_results_df <- do.call(
    what = rbind,
    args = unlist(
      x = lapply(
        X = 1:n_model,
        FUN = function(i) {
          lapply(
            X = 1:n_subject,
            FUN = function(j) {
              # 局部变量，缩短引用路径
              res <- x[[i]][[j]]
              
              # 构建单行 data.frame
              # 注意：这里假设 .flatten_params 现在返回的是 base data.frame
              data.frame(
                fit_model = res@input@settings@name,
                Subject   = res@input@subid,
                ACC       = res@sumstat@ACC,
                LogL      = res@sumstat@LL,
                AIC       = res@sumstat@AIC,
                BIC       = res@sumstat@BIC,
                LogPr     = res@sumstat@LPr,
                LogPo     = res@sumstat@LPo,
                # 将参数部分直接作为列合并进来
                .flatten_params(res@input@params@free),
                stringsAsFactors = FALSE,
                check.names = FALSE
              )
            }
          )
        }
      ),
      recursive = FALSE # 仅解压一层，将 list of lists 变为 list of data.frames
    )
  )
  
  return(final_results_df)
}
