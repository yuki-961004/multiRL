run_m <- function(
    data,
    id = NULL,
    colnames = list(),
    behrule = list(),
    funcs = list(),
    params = list(),
    priors = list(),
    settings = list(),
    engine = "Cpp",
    ...
) {
  request <- .shell_standardize_run_m(
    data = data,
    id = id,
    colnames = colnames,
    behrule = behrule,
    funcs = funcs,
    params = params,
    priors = priors,
    settings = settings,
    engine = engine,
    extra = list(...)
  )

  cpp_result <- .cpp_shell_run_m(
    object = request$features$object,
    reward = request$features$reward,
    action = request$features$action,
    block = request$features$block,
    trial = request$features$trial,
    idinfo = request$features$idinfo,
    exinfo = request$features$exinfo,
    cue = request$behrule$cue,
    rsp = request$behrule$rsp,
    params = request$params$flat,
    free_names = base::names(request$params$free),
    system = request$settings$system,
    prior_names = request$priors$name,
    prior_types = request$priors$type,
    prior_param1 = request$priors$param1,
    prior_param2 = request$priors$param2,
    prior_active = request$priors$active,
    policy = request$settings$policy,
    name = request$settings$name,
    mode = request$settings$mode,
    estimate = request$settings$estimate
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
    input = request,
    result = cpp_result$result,
    sumstat = fit,
    fit = fit,
    extra = request$extra
  )
  base::class(out) <- c("multiRLcpp_run_m", "multiRLcpp_run", "list")
  out
}

.shell_standardize_run_m <- function(
    data,
    id,
    colnames,
    behrule,
    funcs,
    params,
    priors,
    settings,
    engine,
    extra
) {
  if (!base::identical(engine, "Cpp")) {
    base::stop("multiRLcpp 0.5.0 only supports engine = 'Cpp'.")
  }

  if (base::length(funcs) > 0L) {
    base::stop("multiRLcpp 0.5.0 only supports built-in C++ functions.")
  }

  default_colnames <- list(
    subid = "Subject",
    block = "Block",
    trial = "Trial",
    object = NA_character_,
    reward = NA_character_,
    action = "Action",
    exinfo = NA_character_
  )
  colnames <- utils::modifyList(default_colnames, colnames)

  if (!base::is.null(id)) {
    data <- data[data[[colnames$subid]] %in% id, , drop = FALSE]
  }

  if (base::nrow(data) == 0L) {
    base::stop("No data rows remain after id filtering.")
  }

  if (base::is.null(colnames$block)) {
    data$Block <- 1L
    colnames$block <- "Block"
  }

  if (base::length(colnames$object) == 1L &&
      base::is.na(colnames$object)) {
    colnames$object <- base::grep(
      "^Object_",
      base::names(data),
      value = TRUE
    )
  }

  if (base::length(colnames$reward) == 1L &&
      base::is.na(colnames$reward)) {
    colnames$reward <- base::grep(
      "^Reward_",
      base::names(data),
      value = TRUE
    )
  }

  default_params <- list(
    free = list(),
    fixed = list(
      gamma = 1,
      delta = 0.1,
      epsilon = NA_real_,
      zeta = 0
    ),
    constant = list(
      seed = 123,
      chunk = NULL,
      L = NA_real_,
      penalty = 1,
      Q0 = NaN,
      reset = NaN,
      lapse = 0.01,
      threshold = 1,
      bonus = 0,
      weight = 1,
      capacity = 0,
      sticky = 0
    )
  )
  params <- .shell_modify_params(x = default_params, val = params)

  default_behrule <- list()
  behrule <- utils::modifyList(default_behrule, behrule)

  if (base::is.null(behrule$cue) || base::is.null(behrule$rsp)) {
    base::stop("behrule must contain cue and rsp.")
  }

  default_settings <- list(
    name = "unknown",
    mode = "fitting",
    estimate = "MLE",
    policy = "on",
    system = "RL"
  )
  settings <- utils::modifyList(default_settings, settings)

  object <- base::as.matrix(data[, colnames$object, drop = FALSE])
  reward <- base::as.matrix(data[, colnames$reward, drop = FALSE])
  action <- base::as.character(data[[colnames$action]])
  block <- base::as.integer(data[[colnames$block]])
  trial <- base::as.integer(data[[colnames$trial]])
  subid <- base::as.character(data[[colnames$subid]])

  object[] <- base::as.character(object)
  reward[] <- base::as.numeric(reward)

  idinfo <- base::cbind(
    subid = subid,
    block = base::as.character(block),
    trial = base::as.character(trial)
  )

  if (base::length(colnames$exinfo) == 1L &&
      base::is.na(colnames$exinfo)) {
    exinfo <- base::matrix(
      character(),
      nrow = base::nrow(data),
      ncol = 0L
    )
  } else {
    exinfo <- base::as.matrix(data[, colnames$exinfo, drop = FALSE])
    exinfo[] <- base::as.character(exinfo)
  }

  free_params <- params$free
  std_priors <- .shell_standardize_priors(
    priors = priors,
    params = free_params
  )

  list(
    data = data,
    colnames = colnames,
    behrule = behrule,
    funcs = funcs,
    params = list(
      free = params$free,
      fixed = params$fixed,
      constant = params$constant,
      flat = .shell_flatten_params(params)
    ),
    priors = std_priors,
    settings = settings,
    engine = engine,
    features = list(
      object = object,
      reward = reward,
      action = action,
      block = block,
      trial = trial,
      idinfo = idinfo,
      exinfo = exinfo
    ),
    extra = extra
  )
}

.shell_modify_params <- function(x, val) {
  params <- utils::modifyList(x = x, val = val)

  free_names <- base::names(params$free)

  keep_in_fixed <- base::setdiff(
    x = base::names(params$fixed),
    y = free_names
  )
  params$fixed <- params$fixed[keep_in_fixed]

  keep_in_constant <- base::setdiff(
    x = base::names(params$constant),
    y = free_names
  )
  params$constant <- params$constant[keep_in_constant]

  params
}

.shell_flatten_params <- function(params) {
  flat <- base::c(params$free, params$fixed, params$constant)
  base::unlist(flat, use.names = TRUE)
}
