.plot_fitting <- function(x) {
  
################################ [get info] ####################################
  
  # 取出 rsp 名称（列名）
  rsp <- x[[1]][[1]]$multiRL.model@behrule@rsp
  subid <- x[[1]][[1]]$multiRL.model@input@colnames@subid
  block <- x[[1]][[1]]$multiRL.model@input@colnames@block
  action <- x[[1]][[1]]$multiRL.model@input@colnames@action
  
  # 模型名与被试名
  model_names <- names(x)
  sub_names   <- names(x[[1]]) %||% seq_along(x[[1]])
  
############################## [human ratio] ###################################
  
  data_raw <- list()
  
  for (s in sub_names) {
    data_raw[[s]] <- x[[1]][[s]]$multiRL.model@input@data
  }
  data_raw <- do.call(rbind, data_raw)
  
  col_block <- data_raw[[block]]
  col_action <- data_raw[[action]]
  
  # 1. 统计每个 Block 在四个选项的次数
  tbl <- with(data_raw, table(col_block, col_action))
  
  # 2. 计算比例（同行求和后除以该行总和）
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
    
    n_subj  <- length(ratio_list)
    n_block <- length(ratio_list[[1]])
    n_rows  <- n_subj * n_block
    
    # 数值列（按 rsp 动态生成）
    numeric_cols <- stats::setNames(
      object = replicate(
        n = length(rsp),
        expr = numeric(n_rows),
        simplify = FALSE
      ),
      nm = rsp
    )
    
    # 初始化结果表
    out <- data.frame(
      Subject = integer(n_rows),
      Block   = integer(n_rows),
      numeric_cols
    )
    
    row_id <- 1
    
    for (s in seq_len(n_subj)) {
      for (b in seq_len(n_block)) {
        
        vec <- ratio_list[[s]][[b]]  # numeric vector
        
        out$Subject[row_id] <- s
        out$Block[row_id]   <- b
        
        # 填入所有 rsp 数值
        for (l in seq_along(rsp)) {
          out[[rsp[l]]][row_id] <- vec[l]
        }
        
        row_id <- row_id + 1
      }
    }
    
    out
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
    df <- .expand_ratio(ratio_list, rsp)
    
    df_mean <- stats::aggregate(
      x = df[, rsp, drop = FALSE],
      by = list(Block = df[[block]]),
      FUN = mean
    )
    
    df_mean$model <- m
    
    ratio[[m]] <- df_mean
  }
  
  ratio_robot <- do.call(rbind, ratio)
  
################################# [plot] #######################################
  
  ratio <- rbind(ratio_human, ratio_robot)
  rownames(ratio) <- NULL
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
  p
}
