estimate_mle <- function(
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
  .estimate_mle_request(
    data = data,
    id = id,
    colnames = colnames,
    behrule = behrule,
    funcs = funcs,
    params = params,
    priors = priors,
    settings = settings,
    lower = lower,
    upper = upper,
    control = control,
    extra = list(...)
  )
}

.estimate_mle_request <- function(
    data,
    id,
    colnames,
    behrule,
    funcs,
    params,
    priors,
    settings,
    lower,
    upper,
    control,
    extra
) {
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
  features <- .modify_features(data = data, colnames = colnames)
  control <- .modify_estimate_mle_control(
    control = control,
    lower = lower,
    upper = upper,
    free_names = base::names(params$free)
  )

  cpp_result <- .estimate_mle(
    object = features$object,
    reward = features$reward,
    action = features$action,
    block = features$block,
    trial = features$trial,
    idinfo = features$idinfo,
    exinfo = features$exinfo,
    cue = behrule$cue,
    rsp = behrule$rsp,
    params = params$flat,
    free_names = base::names(params$free),
    system = settings$system,
    prior_names = priors$name,
    prior_types = priors$type,
    prior_param1 = priors$param1,
    prior_param2 = priors$param2,
    prior_active = priors$active,
    policy = settings$policy,
    name = settings$name,
    mode = settings$mode,
    maxeval = control$maxeval,
    algorithm = control$algorithm,
    local_algorithm = control$local_algorithm,
    xtol_rel = control$xtol_rel,
    local_xtol_rel = control$local_xtol_rel,
    seed = control$seed,
    lower_bounds = control$lower_bounds,
    upper_bounds = control$upper_bounds
  )

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
      extra = extra
    ),
    fit = cpp_result$fit,
    estimator = list(
      name = "MLE",
      backend = "nlopt",
      algorithm = control$algorithm,
      global_algorithm = control$algorithm,
      local_algorithm = control$local_algorithm,
      control = control
    ),
    diagnostics = cpp_result$diagnostics
  )
  base::class(out) <- c("multiRLcpp_estimate_mle", "multiRLcpp_run", "list")
  out
}

.modify_estimate_mle_control <- function(control, lower, upper, free_names) {
  default_control <- list(
    algorithm = "GN_MLSL",
    local_algorithm = "LN_BOBYQA",
    maxeval = 10000L,
    xtol_rel = 1e-6,
    local_xtol_rel = 1e-8,
    seed = 1004L
  )
  out <- utils::modifyList(default_control, control)
  out$maxeval <- base::as.integer(out$maxeval[[1L]])
  out$algorithm <- base::as.character(out$algorithm[[1L]])
  out$local_algorithm <- base::as.character(out$local_algorithm[[1L]])
  out$xtol_rel <- base::as.numeric(out$xtol_rel[[1L]])
  out$local_xtol_rel <- base::as.numeric(out$local_xtol_rel[[1L]])
  out$seed <- base::as.integer(out$seed[[1L]])
  out$lower_bounds <- .modify_estimate_mle_bounds(
    bounds = lower,
    free_names = free_names,
    default = -Inf
  )
  out$upper_bounds <- .modify_estimate_mle_bounds(
    bounds = upper,
    free_names = free_names,
    default = Inf
  )
  out
}

.modify_estimate_mle_bounds <- function(bounds, free_names, default) {
  if (base::is.null(bounds)) {
    return(base::rep(default, base::length(free_names)))
  }

  if (base::is.list(bounds) && !base::is.null(base::names(bounds))) {
    return(base::as.numeric(base::unlist(bounds[free_names])))
  }

  base::as.numeric(bounds)
}
