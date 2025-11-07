#' engine_RNN
#'
#' @param data data
#' @param behrule behrule
#' @param colnames colnames
#' 
#' @param model model
#' @param funcs funcs
#' @param priors priors
#' @param settings settings
#' 
#' @param control control
#'
#' @returns RNN

engine_RNN <- function(
    data = data,
    behrule = behrule,
    
    colnames,
    funcs = NULL,
    priors,
    settings = NULL,
    model,
    
    control = control
){
  
################################ [default] #####################################
  
  # 默认列名
  default <- list(
    subid = "Subject", 
    block = "Block", 
    trial = "Trial",
    object = NA_character_, 
    reward = NA_character_, 
    action = "Action"
  )
  colnames <- utils::modifyList(x = default, val = colnames)
  
  # 默认方程
  if (is.null(funcs)) {funcs <- list()}
  default <- list(
    rate_func = multiRL::func_alpha,
    prob_func = multiRL::func_beta,
    util_func = multiRL::func_gamma,
    bias_func = multiRL::func_delta,
    expl_func = multiRL::func_epsilon
  )
  funcs <- utils::modifyList(x = default, val = funcs)
  
  # 默认设置
  default <- list(
    name = paste0("Unknown Model"),
    mode = "simulating",
    estimate = "RNN",
    policy = "on"
  )
  settings <- utils::modifyList(x = default, val = settings)

  # 默认控制
  default = list(
    seed = 123,
    core = 1,

    layer = "GRU",
    info = c(colnames$object, colnames$action),
    units = 128,
    sample = 1000,
    batch_size = 10,
    epochs = 100
  )
  control <- utils::modifyList(x = default, val = control)
  # 解放control中的设定, 变成全局变量
  list2env(control, envir = environment())
  
############################### [Simulate] #####################################
  
  multiRL.env <- estimate_0_ENV(
    data = data,
    behrule = behrule,
    colnames = colnames,
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
