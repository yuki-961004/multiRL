.plot_recovery <- function(
    x, model = NULL, param = NULL, metric = "BIC"
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
  
  output <- list()
  
  simulate <- fit <- value <- NULL
  
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
  
############################ [parameter recovery] ##############################
  
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
  
  output$param <- plot
  
############################## [model recovery] ################################
  
  is.latent <- is.na(x$simulate[[1]][[1]]$multiRL.summary@metrics@LL)
  
  if (is.latent) {
    metric <- "ACC"
    message(
      paste0(
        "Since the learning rules are latent, ",
        "Accuracy (ACC) is employed as the metric for model recovery."
      )
    )
  }
  
  .extract_recovery_table <- function(obj, target_metric) {
    # 遍历 simulate 模型层
    do.call(rbind, lapply(names(obj$recovery), function(sim) {
      # 遍历 fit 模型层
      do.call(rbind, lapply(names(obj$recovery[[sim]]), 
        function(fit) {
          # 获取当前组合下的被试列表（1-n）
          subjects <- obj$recovery[[sim]][[fit]]
          
          # 遍历被试层，提取指定的 metric
          do.call(rbind, lapply(seq_along(subjects), 
            function(id) {
              m_obj <- subjects[[id]]$multiRL.summary@metrics
              
              data.frame(
                simulate = sim, fit = fit, id = id,
                metric = methods::slot(m_obj, target_metric)
              )
              
            }))
        }))
    }))
  }
  
  # 0. 读取指标, 并长转宽
  res <- .extract_recovery_table(x, metric)
  res_wide <- stats::reshape(
    res, idvar = c("simulate", "id"), timevar = "fit", 
    direction = "wide", sep = "_"
  )
  names(res_wide) <- gsub("^metric_", "", names(res_wide))
  
  # 1. 识别最优模型
  best_fit <- apply(res_wide[, model_names], 1, function(x) {
    model_names[which.min(x)]
  })
  
  # 2. 生成原始频数表并转换为数据框
  matrix <- prop.table(table(res_wide$simulate, best_fit), margin = 1)
  matrix <- as.data.frame(matrix)
  names(matrix) <- c("simulate", "fit", "value")
  
  # 3. 等级化模型排列顺序
  matrix$simulate <- factor(matrix$simulate, levels = model_names)
  matrix$fit <- factor(matrix$fit, levels = model_names)
  matrix <- matrix[order(matrix$simulate, matrix$fit), ]
  
  # confusion matrix
  conf_matrix <- matrix
  
  # inversion matrix
  invs_matrix <- matrix
  invs_matrix$value <- stats::ave(
    matrix$value, 
    matrix$fit, 
    FUN = function(x) base::round(x / base::sum(x), 2)
  )
  
  title <- c(
    "Confusion Matrix: P(fit model | simulated model)",
    "Inversion Matrix: P(simulated model | fit model)"
  )
  
  matrix <- list()
  matrix[[1]] <- conf_matrix
  matrix[[2]] <- invs_matrix
  
  plot <- list()
  
  for (i in 1:2) {
    plot[[i]] <- ggplot2::ggplot(
      matrix[[i]], ggplot2::aes(x = simulate, y = fit, fill = value)
    ) +
      ggplot2::geom_tile() +
      ggplot2::scale_fill_gradient2(
        low = "#003161", mid = "#55c186", high = "#f0de36",
        limits = c(0, 1), 
        midpoint = 0.4       
      ) + 
      ggplot2::labs(
        title = title[i],
        x = "simulate model", 
        y = "fit model"
      ) + 
      ggplot2::geom_text(
        ggplot2::aes(label = value), size = 10, color = "white"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        legend.position = "none",
        axis.ticks = ggplot2::element_blank(),
        panel.background = ggplot2::element_rect(fill = "white", color = NA),
        plot.background = ggplot2::element_rect(fill = "white", color = NA),
        plot.margin = ggplot2::margin(t = 10, r = 10, b = 10, l = 10),
        plot.title = ggplot2::element_text(
          size = 25, hjust = 1, margin = ggplot2::margin(b = 20)
        ),
        axis.title.y = ggplot2::element_text(
          size = 25, margin = ggplot2::margin(r = 20)
        ),
        axis.title.x = ggplot2::element_text(
          size = 25, margin = ggplot2::margin(t = 20)
        ),
        axis.text = ggplot2::element_text(
          color = "black", size = 20
        )
      )
    print(plot[[i]])
  }
  
  names(plot)[1] <- "Confusion"
  names(plot)[2] <- "Inversion"
  output$model <- plot
  
  invisible(list(param = output$param, model = output$model))
}
