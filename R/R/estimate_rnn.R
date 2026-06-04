estimate_rnn <- function(
    data,
    id = NULL,
    colnames = list(),
    behrule = list(),
    funcs = list(),
    params = list(),
    priors = list(),
    settings = list(),
    lower = NULL,
    upper = NULL,
    control = list(),
    ...
) {
  if (!base::requireNamespace("reticulate", quietly = TRUE)) {
    base::stop(
      "estimate_rnn requires the R package reticulate and Python Keras.",
      call. = FALSE
    )
  }

  colnames <- .modify_colnames(data = data, colnames = colnames)
  data <- .modify_data_id(
    data = data,
    id = id,
    subid = colnames$subid
  )
  params <- .modify_params(params = params)
  priors <- .modify_priors(
    priors = priors,
    params = params$free
  )
  behrule <- .modify_behrule(behrule = behrule)
  settings <- .modify_settings(settings = settings)
  settings$estimate <- "RNN"
  features <- .modify_features(data = data, colnames = colnames)
  control <- .modify_estimate_rnn_control(control = control)

  if (!base::is.null(control$python)) {
    reticulate::use_python(control$python, required = TRUE)
  }
  if (!base::is.null(control$condaenv)) {
    reticulate::use_condaenv(control$condaenv, required = TRUE)
  }

  py_result <- base::tryCatch(
    {
      py <- reticulate::import("multiRL", convert = TRUE)
      py$estimate_rnn(
        object = .matrix_to_row_list(features$object),
        reward = .numeric_matrix_to_row_list(features$reward),
        action = base::as.list(features$action),
        block = base::as.list(features$block),
        trial = base::as.list(features$trial),
        cue = base::as.list(behrule$cue),
        rsp = base::as.list(behrule$rsp),
        params = base::as.list(params$flat),
        free_names = base::as.list(base::names(params$free)),
        system = base::as.list(settings$system),
        policy = settings$policy,
        name = settings$name,
        mode = settings$mode,
        lower = base::as.list(lower),
        upper = base::as.list(upper),
        control = control
      )
    },
    error = function(error) {
      base::stop(
        paste0(
          "estimate_rnn requires a Python environment with multiRL, ",
          "numpy, and Keras/keras3. Use control$python or ",
          "control$condaenv to select that environment. Original error: ",
          base::conditionMessage(error)
        ),
        call. = FALSE
      )
    }
  )

  fit <- base::as.data.frame(py_result$fit, stringsAsFactors = FALSE)
  out <- list(
    input = list(
      data = data,
      colnames = colnames,
      behrule = behrule,
      funcs = funcs,
      params = params,
      priors = priors,
      settings = settings,
      lower = lower,
      upper = upper,
      control = control,
      features = features,
      extra = list(...)
    ),
    fit = fit,
    estimator = py_result$estimator,
    diagnostics = py_result$diagnostics
  )
  base::class(out) <- c("multiRLcpp_estimate_rnn", "multiRLcpp_run", "list")
  out
}

.modify_estimate_rnn_control <- function(control) {
  default_control <- list(
    n_draws = 1000L,
    epochs = 20L,
    batch_size = 32L,
    validation_split = 0.2,
    units = 32L,
    layers = 1L,
    dropout = 0,
    learning_rate = 0.001,
    seed = 123L,
    threads = 0L,
    model_type = "gru",
    verbose = 0L,
    python = NULL,
    condaenv = NULL
  )
  out <- utils::modifyList(default_control, control)
  out$n_draws <- base::as.integer(out$n_draws[[1L]])
  out$epochs <- base::as.integer(out$epochs[[1L]])
  out$batch_size <- base::as.integer(out$batch_size[[1L]])
  out$validation_split <- base::as.numeric(out$validation_split[[1L]])
  out$units <- base::as.integer(out$units[[1L]])
  out$layers <- base::as.integer(out$layers[[1L]])
  out$dropout <- base::as.numeric(out$dropout[[1L]])
  out$learning_rate <- base::as.numeric(out$learning_rate[[1L]])
  out$seed <- base::as.integer(out$seed[[1L]])
  out$threads <- base::as.integer(out$threads[[1L]])
  out$model_type <- base::as.character(out$model_type[[1L]])
  out$verbose <- base::as.integer(out$verbose[[1L]])
  out
}

.matrix_to_row_list <- function(x) {
  base::lapply(
    seq_len(base::nrow(x)),
    function(row) base::as.list(base::as.character(x[row, ]))
  )
}

.numeric_matrix_to_row_list <- function(x) {
  base::lapply(
    seq_len(base::nrow(x)),
    function(row) base::as.list(base::as.numeric(x[row, ]))
  )
}
