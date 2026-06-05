rcv_d <- function(
    data,
    estimator = "abc",
    id = NULL,
    colnames = list(),
    behrule = list(),
    generating = NULL,
    candidates = NULL,
    models = NULL,
    funcs = list(),
    params = list(),
    priors = list(),
    settings = list(),
    lower = NULL,
    upper = NULL,
    lowers = NULL,
    uppers = NULL,
    control = list(),
    ...
) {
  extra <- list(...)
  if (!base::is.null(extra$fit_control)) {
    control <- utils::modifyList(
      x = control,
      val = extra$fit_control,
      keep.null = TRUE
    )
    extra$fit_control <- NULL
    base::warning(
      "rcv_d() now uses control for both sampling and fitting settings.",
      call. = FALSE
    )
  }
  if (!base::is.null(extra$sampler_control)) {
    control <- utils::modifyList(
      x = control,
      val = extra$sampler_control,
      keep.null = TRUE
    )
    extra$sampler_control <- NULL
    base::warning(
      "rcv_d() now uses control for both sampling and fitting settings.",
      call. = FALSE
    )
  }

  if (!base::is.null(lowers)) {
    lower <- lowers
  }
  if (!base::is.null(uppers)) {
    upper <- uppers
  }

  estimator <- .modify_fit_p_estimator(estimator)
  colnames <- .modify_colnames(data = data, colnames = colnames)
  behrule <- .modify_behrule(behrule = behrule)
  data <- .rcv_d_template_data(
    data = data,
    id = id,
    subid = colnames$subid
  )

  generating <- .modify_rcv_d_models(
    models = generating,
    fallback_models = models,
    funcs = funcs,
    params = params,
    priors = priors,
    settings = settings,
    lower = lower,
    upper = upper
  )
  if (base::is.null(candidates)) {
    candidates <- generating
  } else {
    candidates <- .modify_rcv_d_models(
      models = candidates,
      fallback_models = NULL,
      funcs = funcs,
      params = params,
      priors = priors,
      settings = settings,
      lower = lower,
      upper = upper
    )
  }

  control <- .modify_rcv_d_control(
    control = control,
    estimator = estimator
  )

  simulation_parts <- list()
  truth_parts <- list()
  fit_parts <- list()
  raw_fits <- list()

  for (gen_index in seq_along(generating)) {
    gen_spec <- generating[[gen_index]]
    gen_name <- .rcv_d_model_name(gen_spec, gen_index, "generating")
    simulated <- .rcv_d_simulate_model(
      data = data,
      colnames = colnames,
      behrule = behrule,
      spec = gen_spec,
      name = gen_name,
      control = control
    )
    simulation_parts[[gen_name]] <- simulated$simulation
    truth_parts[[gen_name]] <- simulated$truth

    fit_data <- .rcv_d_fit_data(
      simulation = simulated$simulation,
      colnames = colnames
    )

    for (cand_index in seq_along(candidates)) {
      cand_spec <- candidates[[cand_index]]
      cand_name <- .rcv_d_model_name(cand_spec, cand_index, "candidate")
      local_control <- .rcv_d_fit_control(
        estimator = estimator,
        scope = control$scope,
        control = control
      )

      fit_result <- fit_p(
        data = fit_data,
        estimator = estimator,
        id = NULL,
        colnames = colnames,
        behrule = behrule,
        funcs = cand_spec$funcs,
        params = cand_spec$params,
        priors = cand_spec$priors,
        settings = utils::modifyList(
          x = cand_spec$settings,
          val = list(name = cand_name)
        ),
        lower = cand_spec$lower,
        upper = cand_spec$upper,
        control = local_control
      )

      key <- paste(gen_name, cand_name, sep = "::")
      raw_fits[[key]] <- fit_result
      fit_parts[[key]] <- .rcv_d_fit_table(
        fit = fit_result$fit,
        generating_model = gen_name,
        candidate_model = cand_name,
        estimator = estimator
      )
    }
  }

  simulation <- .rcv_d_rbind(simulation_parts)
  truth <- .rcv_d_rbind(truth_parts)
  fit <- .rcv_d_rbind(fit_parts)
  recovery <- .rcv_d_recovery_table(
    truth = truth,
    fit = fit,
    estimator = estimator
  )
  model_recovery <- .rcv_d_model_recovery_table(fit = fit)

  out <- list(
    input = list(
      data = data,
      colnames = colnames,
      behrule = behrule,
      generating = generating,
      candidates = candidates,
      estimator = estimator,
      control = control,
      extra = extra
    ),
    simulation = simulation,
    truth = truth,
    fit = fit,
    recovery = recovery,
    model_recovery = model_recovery,
    raw = raw_fits,
    estimator = list(
      name = base::toupper(estimator),
      shell = "rcv_d",
      scope = if (estimator %in% c("abc", "rnn")) control$scope else NULL
    ),
    diagnostics = list(
      sampler = list(
        n_draws = control$n_draws,
        seed = control$seed,
        threads = control$threads
      ),
      fit_p = list(
        n_fits = base::length(raw_fits),
        estimator = estimator
      ),
      recovery = list(
        n_generating = base::length(generating),
        n_candidates = base::length(candidates)
      )
    )
  )
  base::class(out) <- c("multiRLcpp_rcv_d", "multiRLcpp_run", "list")
  out
}

.rcv_d_template_data <- function(data, id, subid) {
  if (base::is.null(id)) {
    id <- base::unique(data[[subid]])[[1L]]
  }
  data[data[[subid]] == id[[1L]], , drop = FALSE]
}

.modify_rcv_d_models <- function(
    models,
    fallback_models,
    funcs,
    params,
    priors,
    settings,
    lower,
    upper
) {
  if (base::is.null(models)) {
    models <- fallback_models
  }
  if (base::is.null(models)) {
    models <- list(list())
  }
  if (!base::is.list(models) || base::is.null(models[[1L]])) {
    models <- list(models)
  }

  n_models <- base::length(models)

  base::lapply(seq_along(models), function(index) {
    spec <- models[[index]]
    if (base::is.function(spec)) {
      spec <- spec()
    }
    if (!base::is.list(spec)) {
      spec <- list(model = spec)
    }
    indexed <- .rcv_d_indexed_fallback(
      index = index,
      n_models = n_models,
      funcs = funcs,
      params = params,
      priors = priors,
      settings = settings,
      lower = lower,
      upper = upper
    )
    list(
      model = spec$model,
      funcs = .rcv_d_value_override(spec, "funcs", indexed$funcs),
      params = .rcv_d_value(spec, "params", indexed$params),
      priors = .rcv_d_value_override(spec, "priors", indexed$priors),
      settings = .rcv_d_value_override(spec, "settings", indexed$settings),
      lower = .rcv_d_value_override(spec, "lower", indexed$lower),
      upper = .rcv_d_value_override(spec, "upper", indexed$upper)
    )
  })
}

.rcv_d_indexed_fallback <- function(
    index,
    n_models,
    funcs,
    params,
    priors,
    settings,
    lower,
    upper
) {
  list(
    funcs = .rcv_d_indexed_value(funcs, index, n_models),
    params = .rcv_d_indexed_value(params, index, n_models),
    priors = .rcv_d_indexed_value(priors, index, n_models),
    settings = .rcv_d_indexed_value(settings, index, n_models),
    lower = .rcv_d_indexed_value(lower, index, n_models),
    upper = .rcv_d_indexed_value(upper, index, n_models)
  )
}

.rcv_d_indexed_value <- function(value, index, n_models) {
  if (!base::is.list(value)) {
    return(value)
  }
  if (base::length(value) != n_models) {
    return(value)
  }
  if (base::length(base::intersect(
    base::names(value),
    c("free", "fixed", "constant", "type")
  )) > 0L) {
    return(value)
  }
  value[[index]]
}

.rcv_d_value <- function(spec, key, fallback) {
  if (!base::is.null(spec[[key]])) {
    return(spec[[key]])
  }
  fallback
}

.rcv_d_value_override <- function(spec, key, fallback) {
  if (!base::is.null(fallback) && base::length(fallback) > 0L) {
    return(fallback)
  }
  .rcv_d_value(spec, key, fallback)
}

.rcv_d_model_name <- function(spec, index, prefix) {
  if (!base::is.null(spec$settings$name)) {
    return(base::as.character(spec$settings$name[[1L]]))
  }
  paste0(prefix, "_", index)
}

.modify_rcv_d_control <- function(control, estimator) {
  default_scope <- if (estimator %in% c("abc", "rnn")) "shared" else NULL
  out <- utils::modifyList(
    x = list(
      n_draws = 30L,
      seed = 123L,
      threads = 0L,
      scope = default_scope
    ),
    val = control,
    keep.null = TRUE
  )
  out$n_draws <- base::as.integer(out$n_draws[[1L]])
  out$seed <- base::as.integer(out$seed[[1L]])
  out$threads <- base::as.integer(out$threads[[1L]])
  if (!base::is.null(out$scope)) {
    out$scope <- base::tolower(base::as.character(out$scope[[1L]]))
  }
  out
}

.rcv_d_fit_control <- function(estimator, scope, control) {
  if (!estimator %in% c("abc", "rnn")) {
    return(control)
  }
  utils::modifyList(
    x = list(scope = scope),
    val = control,
    keep.null = TRUE
  )
}

.rcv_d_simulate_model <- function(data, colnames, behrule, spec, name, control) {
  params <- .modify_params(params = spec$params)
  priors <- .modify_priors(
    priors = spec$priors,
    params = params$free
  )
  settings <- .modify_settings(spec$settings)
  settings$name <- name
  settings$mode <- "simulating"
  settings$policy <- "on"
  features <- .modify_features(data = data, colnames = colnames)

  lower_bounds <- .modify_estimate_mle_bounds(
    bounds = spec$lower,
    free_names = base::names(params$free),
    default = -Inf
  )
  upper_bounds <- .modify_estimate_mle_bounds(
    bounds = spec$upper,
    free_names = base::names(params$free),
    default = Inf
  )

  cpp <- .shell_rcv_d(
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
    generating_model = name,
    n_draws = control$n_draws,
    seed = control$seed,
    threads = control$threads,
    lower_bounds = lower_bounds,
    upper_bounds = upper_bounds
  )

  list(
    simulation = .rcv_d_name_simulation(cpp$simulation, colnames),
    truth = cpp$truth,
    metadata = cpp$metadata
  )
}

.rcv_d_name_simulation <- function(simulation, colnames) {
  out <- simulation
  object_cols <- grep("^object_[0-9]+$", base::names(out), value = TRUE)
  reward_cols <- grep("^reward_[0-9]+$", base::names(out), value = TRUE)

  for (index in seq_along(object_cols)) {
    if (index <= base::length(colnames$object)) {
      base::names(out)[base::names(out) == object_cols[index]] <-
        colnames$object[index]
    }
  }
  for (index in seq_along(reward_cols)) {
    if (index <= base::length(colnames$reward)) {
      base::names(out)[base::names(out) == reward_cols[index]] <-
        colnames$reward[index]
    }
  }
  out
}

.rcv_d_fit_data <- function(simulation, colnames) {
  out <- simulation
  out[[colnames$subid]] <- out$draw
  out[[colnames$block]] <- out$block
  out[[colnames$trial]] <- out$trial
  out[[colnames$action]] <- out$action
  out
}

.rcv_d_fit_table <- function(fit, generating_model, candidate_model, estimator) {
  out <- base::as.data.frame(fit, stringsAsFactors = FALSE)
  out$generating_model <- generating_model
  out$candidate_model <- candidate_model
  out$estimator <- estimator
  out
}

.rcv_d_recovery_table <- function(truth, fit, estimator) {
  if (base::nrow(truth) == 0L || base::nrow(fit) == 0L) {
    return(data.frame())
  }

  key_cols <- c("generating_model", "draw")
  param_cols <- base::setdiff(
    base::intersect(base::names(truth), base::names(fit)),
    c(key_cols, "subid", "candidate_model", "estimator")
  )
  rows <- list()
  cursor <- 1L

  for (row in seq_len(base::nrow(fit))) {
    matched <- truth[
      truth$generating_model == fit$generating_model[row] &
        truth$draw == fit$subid[row],
      ,
      drop = FALSE
    ]
    if (base::nrow(matched) == 0L) {
      next
    }
    for (parameter in param_cols) {
      true_value <- base::as.numeric(matched[[parameter]][1L])
      recovered <- base::as.numeric(fit[[parameter]][row])
      rows[[cursor]] <- data.frame(
        generating_model = fit$generating_model[row],
        candidate_model = fit$candidate_model[row],
        draw = fit$subid[row],
        subid = fit$subid[row],
        parameter = parameter,
        true = true_value,
        recovered = recovered,
        error = recovered - true_value,
        abs_error = base::abs(recovered - true_value),
        estimator = estimator,
        stringsAsFactors = FALSE
      )
      cursor <- cursor + 1L
    }
  }

  .rcv_d_rbind(rows)
}

.rcv_d_model_recovery_table <- function(fit) {
  if (base::nrow(fit) == 0L) {
    return(data.frame())
  }
  score <- .rcv_d_score(fit)
  out <- data.frame(
    generating_model = fit$generating_model,
    candidate_model = fit$candidate_model,
    draw = fit$subid,
    subid = fit$subid,
    score = score,
    selected = FALSE,
    stringsAsFactors = FALSE
  )

  groups <- base::unique(base::paste(out$generating_model, out$draw))
  for (group in groups) {
    index <- base::which(base::paste(out$generating_model, out$draw) == group)
    if (base::length(index) > 0L) {
      score <- out$score[index]
      if (base::all(!base::is.finite(score))) {
        out$selected[index[1L]] <- TRUE
      } else {
        score[!base::is.finite(score)] <- -Inf
        out$selected[index[base::which.max(score)]] <- TRUE
      }
    }
  }
  out
}

.rcv_d_score <- function(fit) {
  if ("LogPo" %in% base::names(fit)) {
    score <- base::as.numeric(fit$LogPo)
    if (base::any(base::is.finite(score))) {
      return(score)
    }
  }
  if ("LogL" %in% base::names(fit)) {
    score <- base::as.numeric(fit$LogL)
    if (base::any(base::is.finite(score))) {
      return(score)
    }
  }
  if ("NLL" %in% base::names(fit)) {
    score <- -base::as.numeric(fit$NLL)
    if (base::any(base::is.finite(score))) {
      return(score)
    }
  }
  if ("AIC" %in% base::names(fit)) {
    score <- -base::as.numeric(fit$AIC)
    if (base::any(base::is.finite(score))) {
      return(score)
    }
  }
  base::rep(NA_real_, base::nrow(fit))
}

.rcv_d_rbind <- function(parts) {
  parts <- parts[base::vapply(parts, base::NROW, integer(1L)) > 0L]
  if (base::length(parts) == 0L) {
    return(data.frame())
  }
  names <- base::unique(base::unlist(base::lapply(parts, base::names)))
  parts <- base::lapply(parts, function(part) {
    missing <- base::setdiff(names, base::names(part))
    for (name in missing) {
      part[[name]] <- NA
    }
    part[names]
  })
  base::do.call(rbind, parts)
}
