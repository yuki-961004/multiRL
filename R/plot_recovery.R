.plot_recovery <- function(
    x, model = NULL, param = NULL
){
  if (is.null(model)) {
    model_names <- names(x$simulate)
  } else {
    model_names <- model
  }
  
  free_params_names <- list()
  if (is.null(param)) {
    for (i in model_names) {
      free_params_names[[i]] <- names(
        x$simulate[[i]][[1]]$multiRL.summary@params@free
      )
    }
  } else {
    for (i in model_names) {
      free_params_names[[i]] <- param
    }
  }
  
########################## [get free params name] ##############################
  
  .get_replay_params <- function(replay) {
    
    model_names <- names(replay)
    sub_names <- names(replay[[1]]) %||% 1:length(replay[[1]])
    
    free_params <- list()
    
    for (i in model_names) {
      for (j in sub_names) {
        free_params[[i]][[j]] <- unlist(
          replay[[i]][[j]]$multiRL.model@input@params@free
        )
        free_params[[i]][[j]] <- c(Subject = j, free_params[[i]][[j]])
      }
      free_params[[i]] <- as.data.frame(do.call(rbind, free_params[[i]]))
    }
    
    return(free_params)
  }
  
################################## [plot] ######################################
  
  plot <- list()

  for (i in model_names) {
    
    plot[[i]] <- list()

    for (j in free_params_names[[i]]) {
      
      true <- as.numeric(
        .get_replay_params(replay = x$simulate)[[i]][[j]]
      )
      pred <- as.numeric(
        .get_replay_params(replay = x$recovery[[i]])[[i]][[j]]
      )
      
      if (grepl(pattern = "beta", x = j)) {
        true <- log(true)
        pred <- log(pred)
      }
      
      # 1. 确定统一的坐标轴范围
      # 找到 true 和 pred 中的最小值和最大值
      min_val <- base::min(true, pred)
      max_val <- base::max(true, pred)
      limit   <- base::c(min_val, max_val)
      
      p <- data.frame(true = true, pred = pred) |>
        ggplot2::ggplot(
          # aes() 指定了哪些数据列映射到图形属性
          mapping = ggplot2::aes(x = true, y = pred)
        ) +
        # geom_point() 添加散点图层
        ggplot2::geom_point(
          color = "#053562", size = 2
        ) +
        # geom_abline() 添加 1:1 对角线 (理想拟合线)
        ggplot2::geom_abline(
          intercept = 0, slope = 1,
          color = "#55c186", linetype = "dashed", linewidth = 1
        ) +
        # coord_fixed() 强制 X 轴和 Y 轴的单位长度比例为 1:1
        # 结合 scale_x/y_continuous 确保了图形为正方形
        ggplot2::coord_fixed(
          ratio = 1
        ) +
        # scale_x/y_continuous 设置统一的轴范围
        ggplot2::scale_x_continuous(
          name  = "Simulated",
          limit = limit
        ) +
        ggplot2::scale_y_continuous(
          name  = "Fitted",
          limit = limit
        ) +
        # labs() 设置标题和标签
        ggplot2::labs(
          title = paste0(
            i, ": ", j, " (r = ", round(stats::cor(true, pred), 2), ")"
          )
        ) +
        .theme_apa()
      print(p)
      plot[[i]][[j]] <- p
    }
  }
  invisible(plot)
}
