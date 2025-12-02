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
