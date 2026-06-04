estimate_abc <- function(
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
  .estimate_abc_request(
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

.estimate_abc_request <- function(
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
  settings$estimate <- "ABC"
  features <- .modify_features(data = data, colnames = colnames)
  control <- .modify_estimate_abc_control(
    control = control,
    lower = lower,
    upper = upper,
    free_names = base::names(params$free)
  )

  cpp_result <- .estimate_abc(
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
    samples = control$samples,
    tol = control$tol,
    method = control$method,
    reduction = control$reduction,
    n_comp = control$n_comp,
    fake_block = control$fake_block,
    scope = control$scope,
    seed = control$seed,
    threads = control$threads,
    print_level = control$print_level,
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
    estimator = cpp_result$estimator,
    diagnostics = cpp_result$diagnostics
  )
  base::class(out) <- c("multiRLcpp_estimate_abc", "multiRLcpp_run", "list")
  out
}

.modify_estimate_abc_control <- function(control, lower, upper, free_names) {
  default_control <- list(
    samples = 1000L,
    tol = 0.1,
    method = "rejection",
    reduction = "none",
    n_comp = 0L,
    fake_block = 0L,
    scope = "individual",
    seed = 123L,
    threads = 0L,
    print_level = 1L
  )
  out <- utils::modifyList(default_control, control)
  if (!base::is.null(out$reduce)) {
    out$reduction <- out$reduce
  }
  out$samples <- base::as.integer(out$samples[[1L]])
  out$tol <- base::as.numeric(out$tol[[1L]])
  out$method <- base::as.character(out$method[[1L]])
  out$reduction <- base::as.character(out$reduction[[1L]])
  out$n_comp <- base::as.integer(out$n_comp[[1L]])
  out$fake_block <- base::as.integer(out$fake_block[[1L]])
  out$scope <- base::tolower(base::as.character(out$scope[[1L]]))
  supported_scope <- c("individual", "shared", "universal")
  if (!out$scope %in% supported_scope) {
    base::stop(
      paste0(
        "Unknown ABC scope. Supported scopes are: ",
        base::paste(supported_scope, collapse = ", "),
        "."
      ),
      call. = FALSE
    )
  }
  out$seed <- base::as.integer(out$seed[[1L]])
  out$threads <- base::as.integer(out$threads[[1L]])
  out$print_level <- base::as.integer(out$print_level[[1L]])
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
