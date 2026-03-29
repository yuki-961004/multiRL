#' @title
#' The Engine of Recurrent Neural Network (RNN)
#' @name engine_RNN
#' @description
#'  Because TensorFlow requires numeric arrays and input parameters to learn
#'    the mapping between them when building a Recurrent Neural Network (RNN)
#'    model, this function transforms simulated data into a standardized
#'    dataset and invokes TensorFlow to train the model.
#'
#' @param data
#'  A data frame in which each row represents a single trial,
#'    see \link[multiRL]{data}
#' @param colnames
#'  Column names in the data frame,
#'    see \link[multiRL]{colnames}
#' @param behrule
#'  The agent's implicitly formed internal rule,
#'    see \link[multiRL]{behrule}
#' @param model
#'  Reinforcement Learning Model
#' @param funcs
#'  The functions forming the reinforcement learning model,
#'    see \link[multiRL]{funcs}
#' @param priors
#'  Prior probability density function of the free parameters,
#'    see \link[multiRL]{priors}
#' @param settings
#'  Other model settings,
#'    see \link[multiRL]{settings}
#' @param control
#'  Settings manage various aspects of the iterative process,
#'    see \link[multiRL]{control}
#' @param ...
#'  Additional arguments passed to internal functions.
#'
#' @return A specialized TensorFlow-trained Recurrent Neural Network (RNN) object.
#'  The model can be used with the \code{predict()} function to make predictions
#'  on a new data frame, estimating the input parameters that are most likely
#'  to have generated the given dataset.
#'
engine_RNN3 <- function(
  data,
  colnames,
  behrule,

  model,
  funcs = NULL,
  priors,

  settings = NULL,
  control = control,
  ...
) {
  # 确保训练模型和参数恢复时没有用同一份数据
  control$seed <- control$seed * 2
  # 训练模型的样本量是train, 检测模型的量是sample
  control$sample <- control$train
  list2env(control, envir = environment())

############################### [Simulate] #####################################

  multiRL.env <- estimate_0_ENV(
    data = data,
    behrule = behrule,
    colnames = colnames,
    funcs = funcs,
    settings = settings,
  )

  list_simulated <- estimate_2_SBI(
    model = model,
    env = multiRL.env,
    priors = priors,
    control = control
  )

  ################################ [matrix] ######################################

  n_sample <- sample
  n_trials <- nrow(data)
  n_params <- length(priors)

  # 动态获取拆分后的列名 (匹配原列名或拆分衍生的 _1, _2 等)
  mat_cols <- colnames(list_simulated[[1]]$matrix)
  split_info <- unlist(lapply(info, function(col) {
    grep(paste0("^", col, "(_[0-9]+)?$"), mat_cols, value = TRUE)
  }))
  n_info <- length(split_info)

  # Input: n_sample, n_trials, n_info
  X <- array(NA, dim = c(n_sample, n_trials, n_info))

  for (i in 1:n_sample) {
    X[i, , ] <- list_simulated[[i]]$matrix[, split_info, drop = FALSE]
  }

  # Output: n_sample, n_params
  Y <- array(NA, dim = c(n_sample, n_params))

  for (i in 1:n_sample) {
    Y[i, ] <- unlist(list_simulated[[i]]$params)
  }

  ############################## [train/valid] ###################################

  train_indices <- 1:floor(0.8 * n_sample)
  valid_indices <- -train_indices

  X_train <- X[train_indices, , , drop = FALSE]
  X_valid <- X[valid_indices, , , drop = FALSE]

  Y_train <- Y[train_indices, , drop = FALSE]
  Y_valid <- Y[valid_indices, , drop = FALSE]

################################# [keras3] #####################################

  keras3::use_backend(backend)

############################# [loss function] ##################################
  
  if (loss == "NLL") {
    units_out <- n_params * 2
    # 拟合均值和对数方差 (Gaussian Negative Log-Likelihood)
    loss_func <- function(y_true, y_pred) {
      # 前 n_params 个节点预测均值
      mu <- y_pred[, 1:n_params, drop = FALSE]
      # 后 n_params 个节点预测对数方差
      log_var <- y_pred[, (n_params + 1):(2 * n_params), drop = FALSE]

      # 计算精度, 确保方差为正
      precision <- keras3::op_exp(-log_var)
      # 负对数似然核心公式
      loss_val <- precision * keras3::op_square(y_true - mu) + log_var

      return(keras3::op_mean(keras3::op_sum(loss_val, axis = -1L)))
    }
  } else if (loss == "QRL") {
    # 预测3个分位数：5%, 50%, 95%
    units_out <- n_params * 3

    # 分位数损失 (Pinball Loss)
    loss_func <- function(y_true, y_pred) {
      q_values <- c(0.05, 0.50, 0.95)
      total_loss <- 0
      for (i in 1:3) {
        # 获取对应分位数的预测节点
        pred <- y_pred[, ((i - 1) * n_params + 1):(i * n_params), drop = FALSE]
        err <- y_true - pred
        # Pinball公式核心: max(q * err, (q - 1) * err)
        total_loss <- total_loss +
          keras3::op_mean(
            keras3::op_maximum(q_values[i] * err, (q_values[i] - 1.0) * err),
            axis = -1L
          )
      }
      return(total_loss)
    }
  } else if (loss == "MDN") {
    # 混合密度网络 (Mixture Density Network), 默认设定K=3个混合成分
    K <- 3L
    units_out <- n_params * K * 3L

    loss_func <- function(y_true, y_pred) {
      # 将预测输出切分并重塑为 (batch_size, n_params, K)
      pi_logits <- keras3::op_reshape(
        y_pred[, 1:(n_params * K), drop = FALSE],
        c(-1L, n_params, K)
      )
      mu <- keras3::op_reshape(
        y_pred[, (n_params * K + 1):(2 * n_params * K), drop = FALSE],
        c(-1L, n_params, K)
      )
      log_var <- keras3::op_reshape(
        y_pred[, (2 * n_params * K + 1):(3 * n_params * K), drop = FALSE],
        c(-1L, n_params, K)
      )

      # 对K个混合成分进行Softmax，得到权重
      mix_weights <- keras3::op_softmax(pi_logits, axis = -1L)

      # 扩展真实Y值的维度以便广播计算 (batch_size, n_params, 1)
      y_true_exp <- keras3::op_expand_dims(y_true, axis = -1L)

      # 计算高斯分布的对数概率
      cst <- 0.5 * log(2 * base::pi)
      log_prob <- -cst -
        0.5 * log_var -
        0.5 * keras3::op_square(y_true_exp - mu) * keras3::op_exp(-log_var)

      # 加上混合权重的对数 log(pi) + log(N(y|mu, sigma))
      log_mix_weights <- keras3::op_log(mix_weights + 1e-8)
      weighted_log_prob <- log_mix_weights + log_prob

      # Log-Sum-Exp 技巧合并K个成分，防止数值溢出
      log_mix_prob <- keras3::op_logsumexp(
        weighted_log_prob,
        axis = -1L
      )

      # 对所有参数求和，再对批次求均值
      return(keras3::op_mean(-keras3::op_sum(log_mix_prob, axis = -1L)))
    }
  } else if (loss == "MAE") {
    units_out <- n_params
    loss_func <- "mean_absolute_error"
  } else if (loss == "HBR") {
    units_out <- n_params
    loss_func <- "huber_loss"
  } else {
    units_out <- n_params
    loss_func <- "mean_squared_error"
  }

################################# [RNN] ########################################
  
  # Initialize Model (sequential decision making)
  RNN <- keras3::keras_model_sequential(input_shape = c(n_trials, n_info))

  # Recurrent Layer
  switch(
    EXPR = layer,
    "RNN" = {
      RNN <- keras3::layer_simple_rnn(
        object = RNN, 
        units = units
      )
    },
    "GRU" = {
      RNN <- keras3::layer_gru(
        object = RNN,
        units = units
      )
    },
    "LSTM" = {
      RNN <- keras3::layer_lstm(
        object = RNN,
        units = units
      )
    },
    "BiRNN" = {
      RNN <- keras3::layer_bidirectional(
        object = RNN,
        layer = keras3::layer_simple_rnn(units = units)
      )
    },
    "BiGRU" = {
      RNN <- keras3::layer_bidirectional(
        object = RNN,
        layer = keras3::layer_gru(units = units)
      )
    },
    "BiLSTM" = {
      RNN <- keras3::layer_bidirectional(
        object = RNN,
        layer = keras3::layer_lstm(units = units)
      )
    },
  )

  # Hidden Layer
  switch(
    EXPR = as.character(L),
    "0" = {
      RNN <- keras3::layer_dense(
        object = RNN,
        units = units / 2,
        activation = "relu",
        kernel_initializer = keras3::initializer_he_normal()
      )
    },
    "1" = {
      RNN <- keras3::layer_dense(
        object = RNN,
        units = units / 2,
        activation = "relu",
        kernel_initializer = keras3::initializer_he_normal(),
        kernel_regularizer = keras3::regularizer_l1(penalty)
      )
    },
    "2" = {
      RNN <- keras3::layer_dense(
        object = RNN,
        units = units / 2,
        activation = "relu",
        kernel_initializer = keras3::initializer_he_normal(),
        kernel_regularizer = keras3::regularizer_l2(penalty)
      )
    }
  )

  RNN <- RNN |>
    # Dropout Layer
    keras3::layer_dropout(rate = dropout) |>
    # Output Layer
    keras3::layer_dense(
      units = units_out,
      activation = "linear"
    )

  # Loss Function
  switch(
    EXPR = loss,
    "MSE" = ,
    "MAE" = ,
    "HBR" = {
      RNN |>
        keras3::compile(
          loss = loss_func,
          optimizer = "adam",
          metrics = c(loss_func)
        )
    },
    "NLL" = ,
    "MDN" = ,
    "QRL" = {
      RNN |>
        keras3::compile(
          loss = loss_func,
          optimizer = "adam"
        )
    }
  )

  # Training RNN Model
  history <- RNN |>
    keras3::fit(
      x = X_train,
      y = Y_train,
      epochs = epochs,
      batch_size = batch_size,
      validation_data = list(X_valid, Y_valid),
      verbose = 0
    )

  return(RNN)
}
