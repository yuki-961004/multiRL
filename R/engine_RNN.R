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
engine_RNN <- function(
    data,
    colnames,
    behrule,
    
    model,
    funcs = NULL,
    priors,
    
    settings = NULL,
    control = control,
    ...
){

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
  n_info   <- length(info)
  n_params <- length(priors)
  
  # Input: n_sample, n_trials, n_info
  X <- array(NA, dim = c(n_sample, n_trials, n_info))
  
  for (i in 1:n_sample) {
    X[i, , ] <- list_simulated[[i]]$matrix[, info, drop = FALSE]
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
  
############################## [tensorflow] ####################################

  set.seed(seed)
  tensorflow::tf$random$set_seed(seed)
  tensorflow::tf$get_logger()$setLevel('ERROR')
  
  if (loss == "NLL") {
    units_out <- n_params * 2
    # 拟合均值和对数方差 (Gaussian Negative Log-Likelihood)
    loss_func <- function(y_true, y_pred) {
      # 前 n_params 个节点预测均值
      mu <- y_pred[, 1:n_params, drop = FALSE]
      # 后 n_params 个节点预测对数方差
      log_var <- y_pred[, (n_params + 1):(2 * n_params), drop = FALSE]
      
      # 计算精度, 确保方差为正
      precision <- keras::k_exp(-log_var)
      # 负对数似然核心公式
      loss_val <- precision * keras::k_square(y_true - mu) + log_var
      
      return(keras::k_mean(keras::k_sum(loss_val, axis = -1L)))
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
        total_loss <- total_loss + keras::k_mean(
          keras::k_maximum(q_values[i] * err, (q_values[i] - 1.0) * err), 
          axis = -1L
        )
      }
      return(total_loss)
    }
  } else {
    units_out <- n_params
    loss_func <- "mean_squared_error"
  }  

  # Initialize Model (sequential decision making)
  RNN <- keras::keras_model_sequential()

  # Recurrent Layer
  switch(
    EXPR = layer, 
    "GRU" = {
      RNN <- keras::layer_gru(
        object = RNN,
        units = units,
        input_shape = c(n_trials, n_info), 
        return_sequences = FALSE, 
      ) 
    },
    "LSTM" = {
      RNN <- keras::layer_lstm(
        object = RNN,
        units = units,
        input_shape = c(n_trials, n_info), 
        return_sequences = FALSE, 
      ) 
    },
  ) |>
    # Hidden Layer
    keras::layer_dense(
      units = units / 2, 
      activation = "relu"
    ) |>
    # Output Layer
    keras::layer_dense(
      units = units_out, 
      activation = "linear"
    ) 

  # Loss Function
  switch(
    EXPR = loss, 
    "MSE" = {
      RNN |> 
        keras::compile(
          loss = loss_func,
          optimizer = "adam",
          metrics = c(loss_func)
        )
    },
    "NLL" = {
      RNN |> 
        keras::compile(
          loss = loss_func,
          optimizer = "adam"
        )
    },
    "QRL" = {
      RNN |> 
        keras::compile(
          loss = loss_func,
          optimizer = "adam"
        )
    }
  ) 

  # Training RNN Model
  history <- RNN |>
    keras::fit(
      x = X_train,
      y = Y_train,
      epochs = epochs,
      batch_size = batch_size,
      validation_data = list(X_valid, Y_valid),
      verbose = 0
    )
  
  return(RNN)
}
