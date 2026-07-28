estimate_mcmc <- function(
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
  .estimate_mcmc_request(
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

.estimate_mcmc_request <- function(
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
  settings$estimate <- "MCMC"
  features <- .modify_features(data = data, colnames = colnames)
  control <- .modify_estimate_mcmc_control(
    control = control,
    lower = lower,
    upper = upper,
    free_names = base::names(params$free)
  )

  cpp_result <- .estimate_mcmc(
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
    generate = settings$generate,
    name = settings$name,
    mode = settings$mode,
    warmup = control$warmup,
    samples = control$samples,
    chains = control$chains,
    thin = control$thin,
    step_size = control$step_size,
    target_accept = control$target_accept,
    max_tree_depth = control$max_tree_depth,
    seed = control$seed,
    algorithm = control$algorithm,
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
  base::class(out) <- c("multiRLcpp_estimate_mcmc", "multiRLcpp_run", "list")
  out
}

.modify_estimate_mcmc_control <- function(
    control,
    lower,
    upper,
    free_names
) {
  default_control <- list(
    algorithm = "nuts",
    warmup = 1000L,
    samples = 1000L,
    chains = 4L,
    thin = 1L,
    step_size = 0.05,
    target_accept = 0.80,
    max_tree_depth = 8L,
    seed = 1004L
  )
  out <- utils::modifyList(default_control, control)
  out$algorithm <- base::as.character(out$algorithm[[1L]])
  out$warmup <- base::as.integer(out$warmup[[1L]])
  out$samples <- base::as.integer(out$samples[[1L]])
  out$chains <- base::as.integer(out$chains[[1L]])
  out$thin <- base::as.integer(out$thin[[1L]])
  out$step_size <- base::as.numeric(out$step_size[[1L]])
  out$target_accept <- base::as.numeric(out$target_accept[[1L]])
  out$max_tree_depth <- base::as.integer(out$max_tree_depth[[1L]])
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
