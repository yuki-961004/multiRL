fit_p <- function(
    data,
    estimator = "mle",
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
  estimator <- .modify_fit_p_estimator(estimator)
  control <- .modify_fit_p_control(control)

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

.modify_fit_p_control <- function(control) {
  if (base::is.null(control)) {
    control <- list()
  }

  out <- utils::modifyList(
    x = list(scope = "individual"),
    val = control
  )
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
  result$estimator$scope <- scope
  result$input$control$scope <- scope
  result$input$settings$estimate <- base::toupper(estimator)
  base::class(result) <- base::unique(c("multiRLcpp_fit_p", base::class(result)))
  result
}
