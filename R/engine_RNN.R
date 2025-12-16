#' engine_RNN
#'
#' @param data 
#'  A data frame in which each row represents a single trial,
#'    see \link[multiRL]{data} 
#' @param colnames 
#'  Column names in the data frame,
#'    see \link[multiRL]{colnames}
#' @param behrule 
#'  The agent’s implicitly formed internal rule,
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

  tensorflow::tf$get_logger()$setLevel('ERROR')
  
  set.seed(123)
  
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
      units = n_params, 
      activation = "linear"
    ) |>
    # Loss Function
    keras::compile(
      loss = "mean_squared_error",
      optimizer = "adam",
      metrics = c("mean_absolute_error")
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
