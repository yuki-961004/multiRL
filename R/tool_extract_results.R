.extract_results <- function(x) {
  
  n_model <- length(x)
  n_subject <- length(x[[1]])
  
  # 必须先定义 .flatten_params 辅助函数
  .flatten_params <- function(free_params) {
    
    # unlist 自动将向量元素命名为 "alpha1", "alpha2"
    param_vector <- unlist(x = free_params, recursive = TRUE)
    
    # tibble::as_tibble_row 避免了创建中间数据框
    return(tibble::as_tibble_row(x = param_vector))
  }
  
  # 外部循环 (i: model)
  final_results_df <- purrr::map_dfr(
    .x = 1:n_model,
    .f = function(i) {
      
      # 内部循环 (j: subject)
      purrr::map_dfr(
        .x = 1:n_subject, 
        .f = function(j) {
          
          # 局部变量，避免重复访问
          model_result <- x[[i]][[j]]
          
          # 提取固定统计量
          stat_df <- dplyr::tibble(
            fit_model = model_result@input@settings@name,
            Subject = model_result@input@subid,
            ACC = model_result@sumstat@ACC,
            LogL = model_result@sumstat@LL,
            AIC = model_result@sumstat@AIC,
            BIC = model_result@sumstat@BIC,
            LogPr = model_result@sumstat@LPr,
            LogPo = model_result@sumstat@LPo
          )
          
          # 提取参数 (params) 并平铺
          params_df <- .flatten_params(
            free_params = model_result@input@params@free
          )
          
          # 合并并返回
          return(dplyr::bind_cols(stat_df, params_df))
        }
      )
    }
  )
  
  return(final_results_df)
}
