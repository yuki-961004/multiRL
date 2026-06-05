fit_p <- function(
    data,
    estimator = "mle",
    id = NULL,
    colnames = list(),
    behrule = list(),
    models = NULL,
    funcs = list(),
    params = list(),
    priors = list(),
    settings = list(),
    lower = NULL,
    upper = NULL,
    control = list(),
    ...
) {
  estimator <- .modify_fit_p_estimator(estimator)
  control <- .modify_fit_p_control(control, estimator)

  if (!base::is.null(models)) {
    return(.fit_p_models(
      data = data,
      estimator = estimator,
      id = id,
      colnames = colnames,
      behrule = behrule,
      models = models,
      funcs = funcs,
      params = params,
      priors = priors,
      settings = settings,
      lower = lower,
      upper = upper,
      control = control,
      extra = list(...)
    ))
  }

  settings <- utils::modifyList(
    x = list(estimate = base::toupper(estimator)),
    val = settings
  )

  result <- switch(
    estimator,
    "mle" = estimate_mle(
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
      ...
    ),
    "map" = estimate_map(
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
      ...
    ),
    "mcmc" = estimate_mcmc(
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
      ...
    ),
    "abc" = estimate_abc(
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
      ...
    ),
    "rnn" = estimate_rnn(
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
      ...
    )
  )

  .tag_fit_p_result(
    result = result,
    estimator = estimator,
    scope = control$scope
  )
}

.modify_fit_p_estimator <- function(estimator) {
  estimator <- base::tolower(base::as.character(estimator[[1L]]))
  supported <- c("mle", "map", "mcmc", "abc", "rnn")
  if (!estimator %in% supported) {
    base::stop(
      paste0(
        "Unknown estimator. Supported estimators in fit_p v0.5.0-12 are: ",
        base::paste(supported, collapse = ", "),
        "."
      ),
      call. = FALSE
    )
  }
  estimator
}

.modify_fit_p_control <- function(control, estimator) {
  if (base::is.null(control)) {
    control <- list()
  }

  default_scope <- if (estimator %in% c("abc", "rnn")) "individual" else NULL
  out <- utils::modifyList(
    x = list(scope = default_scope),
    val = control,
    keep.null = TRUE
  )
  if (base::is.null(out$scope)) {
    return(out)
  }

  out$scope <- base::tolower(base::as.character(out$scope[[1L]]))
  supported <- c("individual", "shared", "universal")
  if (!out$scope %in% supported) {
    base::stop(
      paste0(
        "Unknown scope. Supported scopes in fit_p v0.5.0-12 are: ",
        base::paste(supported, collapse = ", "),
        "."
      ),
      call. = FALSE
    )
  }
  out
}

.tag_fit_p_result <- function(result, estimator, scope) {
  result$estimator$shell <- "fit_p"
  if (estimator %in% c("abc", "rnn")) {
    result$estimator$scope <- scope
    result$input$control$scope <- scope
    scope_info <- result$diagnostics$scope
    if (base::is.null(scope_info)) {
      scope_info <- list(scope = scope)
    }
    scope_info$estimator <- estimator
    scope_info$applies <- TRUE
    result$diagnostics$scope <- scope_info
  } else {
    result$diagnostics$scope <- list(
      estimator = estimator,
      scope = scope,
      applies = FALSE,
      reason = "scope applies only to SBI estimators: abc and rnn"
    )
  }
  result$input$settings$estimate <- base::toupper(estimator)
  base::class(result) <- base::unique(c("multiRLcpp_fit_p", base::class(result)))
  result
}

.fit_p_models <- function(
    data,
    estimator,
    id,
    colnames,
    behrule,
    models,
    funcs,
    params,
    priors,
    settings,
    lower,
    upper,
    control,
    extra
) {
  specs <- .modify_fit_p_models(
    models = models,
    funcs = funcs,
    params = params,
    priors = priors,
    settings = settings,
    lower = lower,
    upper = upper
  )

  raw <- list()
  fit_parts <- list()

  for (index in seq_along(specs)) {
    spec <- specs[[index]]
    model <- .fit_p_model_name(spec, index)
    model_id <- base::paste0(model, "_", index)

    result <- fit_p(
      data = data,
      estimator = estimator,
      id = id,
      colnames = colnames,
      behrule = behrule,
      funcs = spec$funcs,
      params = spec$params,
      priors = spec$priors,
      settings = utils::modifyList(
        x = spec$settings,
        val = list(name = model)
      ),
      lower = spec$lower,
      upper = spec$upper,
      control = control
    )
    result$fit$model <- model
    result$fit$model_id <- model_id
    result$fit <- result$fit[
      c("model", "model_id", base::setdiff(
        base::names(result$fit),
        c("model", "model_id")
      ))
    ]
    raw[[model_id]] <- result
    fit_parts[[model_id]] <- result$fit
  }

  fit <- .fit_p_rbind(fit_parts)
  out <- list(
    input = list(
      data = data,
      colnames = colnames,
      behrule = behrule,
      models = specs,
      estimator = estimator,
      control = control,
      extra = extra
    ),
    fit = fit,
    raw = raw,
    estimator = list(
      name = base::toupper(estimator),
      shell = "fit_p"
    ),
    diagnostics = list(
      n_models = base::length(specs),
      model_id = base::names(raw)
    )
  )
  base::class(out) <- c("multiRLcpp_fit_p", "multiRLcpp_run", "list")
  out
}

.modify_fit_p_models <- function(
    models,
    funcs,
    params,
    priors,
    settings,
    lower,
    upper
) {
  if (!base::is.list(models)) {
    models <- list(models)
  }
  base::lapply(seq_along(models), function(index) {
    spec <- models[[index]]
    if (base::is.function(spec)) {
      spec <- spec()
    }
    if (!base::is.list(spec)) {
      spec <- list(model = spec)
    }
    list(
      model = spec$model,
      process = spec$process,
      funcs = .fit_p_value(spec, "funcs", funcs),
      params = .fit_p_value(spec, "params", params),
      priors = .fit_p_value(spec, "priors", priors),
      settings = .fit_p_value(spec, "settings", settings),
      lower = .fit_p_value(spec, "lower", lower),
      upper = .fit_p_value(spec, "upper", upper)
    )
  })
}

.fit_p_value <- function(spec, key, fallback) {
  if (!base::is.null(spec[[key]])) {
    return(spec[[key]])
  }
  fallback
}

.fit_p_model_name <- function(spec, index) {
  if (!base::is.null(spec$settings$name)) {
    return(base::as.character(spec$settings$name[[1L]]))
  }
  if (!base::is.null(spec$model)) {
    return(base::as.character(spec$model[[1L]]))
  }
  base::paste0("model_", index)
}

.fit_p_rbind <- function(parts) {
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
