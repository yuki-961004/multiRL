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
  colnames <- .modify_colnames(data = data, colnames = colnames)
  data <- .modify_data_id(
    data = data,
    id = id,
    subid = colnames$subid
  )
  params  <- .modify_params(params = params)
  priors  <- .modify_priors(
    priors = priors,
    params = params$free
  )
  behrule  <- .modify_behrule(behrule = behrule)
  settings <- .modify_settings(settings = settings)
  settings$estimate <- "RNN"
  features <- .modify_features(data = data, colnames = colnames)
  control  <- .modify_estimate_rnn_control(control = control)

  cpp_result <- .estimate_rnn_data(
    object         = features$object,
    reward         = features$reward,
    action         = features$action,
    block          = features$block,
    trial          = features$trial,
    idinfo         = features$idinfo,
    exinfo         = features$exinfo,
    cue            = behrule$cue,
    rsp            = behrule$rsp,
    params         = params$flat,
    free_names     = base::names(params$free),
    system         = settings$system,
    prior_names    = priors$name,
    prior_types    = priors$type,
    prior_param1   = priors$param1,
    prior_param2   = priors$param2,
    prior_active   = priors$active,
    policy         = settings$policy,
    name           = settings$name,
    mode           = settings$mode,
    n_draws        = control$n_draws,
    seed           = control$seed,
    threads        = control$threads,
    interop_threads= control$interop_threads,
    epochs         = control$epochs,
    batch_size     = control$batch_size,
    units          = control$units,
    layers         = control$layers,
    dropout        = control$dropout,
    learning_rate  = control$learning_rate,
    architecture   = control$architecture,
    loss           = control$loss,
    regularization = control$regularization,
    penalty        = control$penalty,
    verbose        = control$verbose,
    device         = control$device,
    scope          = control$scope,
    lower_bounds   = .modify_bound_vector(lower, base::names(params$free)),
    upper_bounds   = .modify_bound_vector(upper, base::names(params$free))
  )

  out <- list(
    input = list(
      data     = data,
      colnames = colnames,
      behrule  = behrule,
      funcs    = funcs,
      params   = params,
      priors   = priors,
      settings = settings,
      lower    = lower,
      upper    = upper,
      control  = control,
      features = features,
      extra    = list(...)
    ),
    fit         = cpp_result$fit,
    estimator   = cpp_result$estimator,
    diagnostics = cpp_result$diagnostics
  )
  base::class(out) <- c("multiRLcpp_estimate_rnn", "multiRLcpp_run", "list")
  out
}

.modify_estimate_rnn_control <- function(control) {
  default_control <- list(
    n_draws        = 1000L,
    epochs         = 20L,
    batch_size     = 32L,
    validation_split = 0,
    units          = 32L,
    layers         = 1L,
    dropout        = 0,
    learning_rate  = 0.001,
    seed           = 123L,
    threads        = 0L,
    interop_threads= 0L,
    backend        = "torch",
    loss           = "mse",
    architecture   = "gru",
    regularization = "none",
    penalty        = 0.0,
    device         = "cpu",
    scope          = "individual",
    verbose        = 0L
  )
  out <- utils::modifyList(default_control, control)
  out$n_draws         <- base::as.integer(out$n_draws[[1L]])
  out$epochs          <- base::as.integer(out$epochs[[1L]])
  out$batch_size      <- base::as.integer(out$batch_size[[1L]])
  out$validation_split<- base::as.numeric(out$validation_split[[1L]])
  out$units           <- base::as.integer(out$units[[1L]])
  out$layers          <- base::as.integer(out$layers[[1L]])
  out$dropout         <- base::as.numeric(out$dropout[[1L]])
  out$learning_rate   <- base::as.numeric(out$learning_rate[[1L]])
  out$seed            <- base::as.integer(out$seed[[1L]])
  out$threads         <- base::as.integer(out$threads[[1L]])
  out$interop_threads <- base::as.integer(out$interop_threads[[1L]])
  out$backend         <- base::tolower(base::as.character(out$backend[[1L]]))
  out$loss            <- base::tolower(base::as.character(out$loss[[1L]]))
  out$architecture    <- base::tolower(base::as.character(out$architecture[[1L]]))
  out$regularization  <- base::tolower(base::as.character(out$regularization[[1L]]))
  out$penalty         <- base::as.numeric(out$penalty[[1L]])
  out$scope           <- base::tolower(base::as.character(out$scope[[1L]]))
  out$verbose         <- base::as.integer(out$verbose[[1L]])
  out$device          <- base::tolower(base::as.character(out$device[[1L]]))
  out
}

.modify_bound_vector <- function(bound, free_names) {
  if (base::is.null(bound)) {
    return(base::rep(NA_real_, base::length(free_names)))
  }
  if (base::is.null(base::names(bound))) {
    return(base::as.numeric(bound))
  }
  base::vapply(
    free_names,
    function(name) {
      if (name %in% base::names(bound)) {
        return(base::as.numeric(bound[[name]][[1L]]))
      }
      NA_real_
    },
    FUN.VALUE = numeric(1L)
  )
}
