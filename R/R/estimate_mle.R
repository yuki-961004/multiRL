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
    engine = "Cpp",
    ...
) {
  .estimate_mle(
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
    engine = engine,
    extra = list(...)
  )
}

.estimate_mle <- function(
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
    engine,
    extra
) {
  if (!base::identical(engine, "Cpp")) {
    base::stop("multiRLcpp 0.5.0 only supports engine = 'Cpp'.")
  }

  if (base::length(funcs) > 0L) {
    base::stop("multiRLcpp 0.5.0 only supports built-in C++ functions.")
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
  features <- .modify_features(data = data, colnames = colnames)
  control <- .modify_estimate_mle_control(
    control = control,
    lower = lower,
    upper = upper,
    free_names = base::names(params$free)
  )

  cpp_result <- .cpp_estimate_mle(
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
    estimate = "MLE",
    maxeval = control$maxeval,
    algorithm = control$algorithm,
    xtol_rel = control$xtol_rel,
    lower_bounds = control$lower_bounds,
    upper_bounds = control$upper_bounds
  )

  fit <- base::data.frame(
    ACC = cpp_result$metric$ACC,
    LogL = cpp_result$metric$LogL,
    LogPr = cpp_result$metric$LogPr,
    LogPo = cpp_result$metric$LogPo,
    AIC = cpp_result$metric$AIC,
    BIC = cpp_result$metric$BIC
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
      engine = engine,
      features = features,
      extra = extra
    ),
    params = cpp_result$params,
    fit = fit,
    estimator = cpp_result$estimator
  )
  base::class(out) <- c("multiRLcpp_estimate_mle", "multiRLcpp_run", "list")
  out
}

.modify_estimate_mle_control <- function(control, lower, upper, free_names) {
  default_control <- list(
    algorithm = "LN_BOBYQA",
    maxeval = 10000L,
    xtol_rel = 1e-6
  )
  out <- utils::modifyList(default_control, control)
  out$maxeval <- base::as.integer(out$maxeval[[1L]])
  out$algorithm <- base::as.character(out$algorithm[[1L]])
  out$xtol_rel <- base::as.numeric(out$xtol_rel[[1L]])
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
