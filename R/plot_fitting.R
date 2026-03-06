.plot_fitting <- function(x) {
  
################################ [get info] ####################################
  
  # 取出 rsp 名称（列名）
  rsp     <- x[[1]][[1]]$multiRL.model@behrule@rsp
  subid   <- x[[1]][[1]]$multiRL.model@input@colnames@subid
  block   <- x[[1]][[1]]$multiRL.model@input@colnames@block
  action  <- x[[1]][[1]]$multiRL.model@input@colnames@action
  
  # 模型名与被试名
  model_names <- names(x)
  sub_names   <- names(x[[1]]) %||% seq_along(x[[1]])
  
############################## [human ratio] ###################################
  
  data_raw <- list()
  
  for (s in sub_names) {
    data_raw[[s]] <- x[[1]][[s]]$multiRL.model@input@data
  }
  data_raw <- do.call(rbind, data_raw)
  
  col_block   <- data_raw[[block]]
  col_action  <- data_raw[[action]]
  
  # 1. 统计每个 Block 不同action的次数
  tbl <- with(data_raw, table(col_block, col_action))
  
  # 2. 计算每个 Block 不同action的比率
  ratio_raw <- prop.table(tbl, margin = 1)
  
  # 3. 转成数据框
  ratio_human <- data.frame(
    Block = as.integer(rownames(ratio_raw)),
    as.data.frame.matrix(ratio_raw)
  )
  
  ratio_human$model <- "Human"
  
############################## [expand_ratio] ##################################
  
  # ---- 辅助函数：将 ratio list 展开成 data.frame ----
  .expand_ratio <- function(ratio_list, rsp) {
    
    # 使用 lapply 遍历被试，将每个矩阵转换为 data.frame
    do.call(
      what = rbind,
      args = lapply(
        X   = seq_along(ratio_list),
        FUN = function(s) {
          
          # 提取当前被试的 [Block x Action] 矩阵
          res_mat <- ratio_list[[s]]
          
          # 直接创建包含 Subject 和 Block 的基础表
          # 使用 nrow() 确保 Block 数量正确
          sub_df <- data.frame(
            Subject = as.integer(s),
            Block   = seq_len(nrow(res_mat))
          )
          
          # 将矩阵整列赋值给对应的 Action 列名 (rsp)
          sub_df[, rsp] <- res_mat
          
          return(sub_df)
        }
      )
    )
    
  }
  
############################## [robot ratio] ###################################
  
  # ---- 主过程：从 replay 提取所有模型的 ratio ----
  ratio <- vector("list", length(model_names))
  names(ratio) <- model_names
  
  for (m in model_names) {
    
    # 收集每个模型的被试 → block → ratio
    ratio_list <- vector("list", length(sub_names))
    
    for (s in seq_along(sub_names)) {
      ratio_list[[s]] <- x[[m]][[s]]$multiRL.summary@metrics@ABC$ratio
    }
    
    # 展开成大表
    ratio_df <- .expand_ratio(ratio_list, rsp)
    
    ratio_model <- stats::aggregate(
      x = ratio_df[, rsp, drop = FALSE],
      by = list(Block = ratio_df[[block]]),
      FUN = mean
    )
    
    ratio_model$model <- m
    
    ratio[[m]] <- ratio_model
  }
  
  ratio_robot <- do.call(rbind, ratio)
  
################################# [plot] #######################################
  
  ratio <- rbind(ratio_human, ratio_robot)
  row.names(ratio) <- NULL
  ratio$model <- factor(ratio$model, levels = unique(ratio$model))
  
  long_df <- stats::reshape(
    ratio,
    varying = rsp,
    v.names = "value",
    timevar = "action",
    times = rsp,
    direction = "long"
  )
  
  # 整理 rownames
  row.names(long_df) <- NULL
  
  p <- long_df |>
    ggplot2::ggplot(
      mapping = ggplot2::aes(
        x = long_df$Block,
        y = long_df$value,
        color = long_df$model,
        group = long_df$model
      )
    ) +
    ggplot2::geom_line(linewidth = 1) +
    ggplot2::geom_point(size = 2) +
    ggplot2::facet_wrap(~ action, nrow = 2) +
    ggplot2::scale_x_continuous(breaks = sort(unique(long_df$Block))) +
    ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    ggplot2::scale_color_manual(
      values = stats::setNames(
        object = .palette(length(model_names) + 1), nm = c("Human", model_names)
      )
    ) +
    ggplot2::labs(
      x = "Block",
      y = "Action Ratio",
      color = "Model"
    ) +
    .theme_apa()
  print(p)
  invisible(p)
}
