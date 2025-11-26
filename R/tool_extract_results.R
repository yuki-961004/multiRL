.extract_results <- function(x) {
  
  # 第一步: 提取数据 (Extract)
  # 依然保持 nested lapply 结构，因为这是遍历 list-of-lists 的标准方式
  nested_res <- lapply(x, function(m_list) {
    lapply(m_list, function(res) {
      
      # 提取参数并转置
      # recursive = TRUE 确保即使 params 是嵌套列表也能拉平
      params_vec <- unlist(res@input@params@free, recursive = TRUE)
      
      params_df <- as.data.frame(
        t(params_vec), 
        stringsAsFactors = FALSE
      )
      
      # 提取统计量
      stat_df <- data.frame(
        fit_model = res@input@settings@name,
        Subject   = res@input@subid,
        ACC       = res@sumstat@ACC,
        LogL      = res@sumstat@LL,
        AIC       = res@sumstat@AIC,
        BIC       = res@sumstat@BIC,
        LogPr     = res@sumstat@LPr,
        LogPo     = res@sumstat@LPo,
        stringsAsFactors = FALSE
      )
      
      # 在单次迭代内部，stat_df 和 params_df 行数必定为 1，可以直接 cbind
      return(cbind(stat_df, params_df))
    })
  })
  
  # 第二步: 将 list of lists 转换为 list of data.frames
  flat_list <- unlist(nested_res, recursive = FALSE)
  
  # 第三步: 合并list成dataframe, 约等于dplyr::bind_rows
  final_results_df <- .rbind_fill(flat_list)
  
  return(final_results_df)
}