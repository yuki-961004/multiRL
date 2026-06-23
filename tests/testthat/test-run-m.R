testthat::local_edition(3)

extract_binary_metric <- function(model) {
  utils::capture.output(summary_value <- summary(model))
  metric <- summary_value[[2]]
  c(
    LogL = metric[metric[, "Metric"] == "LogL", "Value"],
    LogPr = metric[metric[, "Metric"] == "LogPr", "Value"],
    LogPo = metric[metric[, "Metric"] == "LogPo", "Value"]
  )
}

extract_legacy_metric <- function(model) {
  utils::capture.output(summary_value <- multiRL::summary(model))
  c(
    LogL = summary_value@metrics@LL,
    LogPr = summary_value@metrics@LPr,
    LogPo = summary_value@metrics@LPo
  )
}

extract_cpp_metric <- function(model) {
  c(
    LogL = model$fit$LogL,
    LogPr = model$fit$LogPr,
    LogPo = model$fit$LogPo
  )
}

convert_legacy_prior <- function(prior) {
  if (base::is.list(prior)) {
    return(prior)
  }
  if (base::is.function(prior)) {
    body_call <- base::body(prior)
    call_list <- base::as.list(body_call)
    func_name <- base::deparse(call_list[[1L]])
    
    if (base::grepl("dbeta", func_name)) {
      matched <- base::match.call(stats::dbeta, body_call)
      shape1 <- if (!base::is.null(matched[["shape1"]])) base::eval(matched[["shape1"]]) else 1
      shape2 <- if (!base::is.null(matched[["shape2"]])) base::eval(matched[["shape2"]]) else 1
      return(base::list(type = "beta", shape1 = shape1, shape2 = shape2))
    } else if (base::grepl("dexp", func_name)) {
      matched <- base::match.call(stats::dexp, body_call)
      rate <- if (!base::is.null(matched[["rate"]])) base::eval(matched[["rate"]]) else 1
      return(base::list(type = "exponential", rate = rate))
    } else if (base::grepl("dnorm", func_name)) {
      matched <- base::match.call(stats::dnorm, body_call)
      mean <- if (!base::is.null(matched[["mean"]])) base::eval(matched[["mean"]]) else 0
      sd <- if (!base::is.null(matched[["sd"]])) base::eval(matched[["sd"]]) else 1
      return(base::list(type = "normal", mean = mean, sd = sd))
    }
  }
  base::stop("Unsupported prior format")
}

convert_legacy_priors <- function(priors) {
  base::lapply(priors, convert_legacy_prior)
}

tab_subject_one <- function() {
  multiRL::TAB[multiRL::TAB[, "Subject"] == 1, ]
}

legacy_tab_run <- function(params, priors, settings) {
  multiRL::run_m(
    engine = "R",
    data = tab_subject_one(),
    behrule = list(
      cue = c("A", "B", "C", "D"),
      rsp = c("A", "B", "C", "D")
    ),
    colnames = list(
      object = c("L_choice", "R_choice"),
      reward = c("L_reward", "R_reward"),
      action = "Sub_Choose"
    ),
    params = params,
    priors = priors,
    settings = settings
  )
}

cpp_tab_run <- function(params, priors, settings) {
  multiRLcpp::run_m(
    data = tab_subject_one(),
    behrule = list(
      cue = c("A", "B", "C", "D"),
      rsp = c("A", "B", "C", "D")
    ),
    colnames = list(
      object = c("L_choice", "R_choice"),
      reward = c("L_reward", "R_reward"),
      action = "Sub_Choose"
    ),
    params = params,
    priors = convert_legacy_priors(priors),
    settings = settings
  )
}

testthat::test_that("0_Benchmark first chunk stays aligned", {
  testthat::skip_if_not_installed("binaryRL")
  testthat::skip_if_not_installed("multiRL")

  subject_one <- binaryRL::Mason_2024_G2[
    binaryRL::Mason_2024_G2[, "Subject"] == 1,
  ]

  binary_model <- binaryRL::run_m(
    mode = "fit",
    data = binaryRL::Mason_2024_G2,
    id = 1,
    n_params = 3,
    n_trials = 360,
    pi = 0.1,
    eta = c(0.3),
    tau = c(0.5),
    threshold = 20,
    priors = list(
      eta = function(x) stats::dbeta(
        x,
        shape1 = 2,
        shape2 = 2,
        log = TRUE
      ),
      tau = function(x) stats::dexp(x, rate = 1, log = TRUE)
    ),
    policy = "off"
  )

  legacy_model <- multiRL::run_m(
    engine = "R",
    data = subject_one,
    behrule = list(
      cue = c("A", "B", "C", "D"),
      rsp = c("A", "B", "C", "D")
    ),
    colnames = list(
      object = c("L_choice", "R_choice"),
      reward = c("L_reward", "R_reward"),
      action = "Sub_Choose"
    ),
    params = list(
      free = list(alpha = 0.3, beta = 0.5),
      fixed = list(threshold = 20)
    ),
    priors = list(
      alpha = function(x) stats::dbeta(
        x,
        shape1 = 2,
        shape2 = 2,
        log = TRUE
      ),
      beta = function(x) stats::dexp(x, rate = 1, log = TRUE)
    ),
    settings = list(
      name = "TD",
      mode = "fitting",
      estimate = "MLE",
      policy = "off"
    )
  )

  cpp_model <- multiRLcpp::run_m(
    data = subject_one,
    behrule = list(
      cue = c("A", "B", "C", "D"),
      rsp = c("A", "B", "C", "D")
    ),
    colnames = list(
      object = c("L_choice", "R_choice"),
      reward = c("L_reward", "R_reward"),
      action = "Sub_Choose"
    ),
    params = list(
      free = list(alpha = 0.3, beta = 0.5),
      fixed = list(threshold = 20)
    ),
    priors = list(
      alpha = list(type = "beta", shape1 = 2, shape2 = 2),
      beta = list(type = "exponential", rate = 1)
    ),
    settings = list(
      name = "TD",
      mode = "fitting",
      estimate = "MLE",
      policy = "off"
    )
  )

  binary_metric <- extract_binary_metric(binary_model)
  legacy_metric <- extract_legacy_metric(legacy_model)
  cpp_metric <- extract_cpp_metric(cpp_model)

  testthat::expect_equal(cpp_metric, legacy_metric, tolerance = 1e-8)
  testthat::expect_equal(cpp_metric, binary_metric, tolerance = 1e-4)
})

testthat::test_that("policy-off deterministic model paths match legacy run_m", {
  testthat::skip_if_not_installed("multiRL")

  scenarios <- list(
    TD = list(
      params = list(
        free = list(alpha = 0.3, beta = 0.5),
        fixed = list(threshold = 20)
      ),
      priors = list(
        alpha = function(x) stats::dbeta(x, 2, 2, log = TRUE),
        beta = function(x) stats::dexp(x, rate = 1, log = TRUE)
      ),
      settings = list(
        name = "TD",
        mode = "fitting",
        estimate = "MLE",
        policy = "off"
      )
    ),
    RSTD_RL = list(
      params = list(
        free = list(alphaN = 0.3, alphaP = 0.7, beta = 0.5),
        fixed = list(threshold = 20)
      ),
      priors = list(
        alphaN = function(x) stats::dbeta(x, 2, 2, log = TRUE),
        alphaP = function(x) stats::dbeta(x, 2, 2, log = TRUE),
        beta = function(x) stats::dexp(x, rate = 1, log = TRUE)
      ),
      settings = list(
        name = "RSTD",
        mode = "fitting",
        estimate = "MLE",
        policy = "off",
        system = "RL"
      )
    ),
    RSTD_WM = list(
      params = list(
        free = list(alphaN = 0.3, alphaP = 0.7, beta = 0.5),
        fixed = list(threshold = 20),
        constant = list(weight = 1)
      ),
      priors = list(
        alphaN = function(x) stats::dbeta(x, 2, 2, log = TRUE),
        alphaP = function(x) stats::dbeta(x, 2, 2, log = TRUE),
        beta = function(x) stats::dexp(x, rate = 1, log = TRUE)
      ),
      settings = list(
        name = "RSTD",
        mode = "fitting",
        estimate = "MLE",
        policy = "off",
        system = "WM"
      )
    ),
    Utility = list(
      params = list(
        free = list(alpha = 0.3, beta = 0.5, gamma = 0.7),
        fixed = list(threshold = 20)
      ),
      priors = list(
        alpha = function(x) stats::dbeta(x, 2, 2, log = TRUE),
        beta = function(x) stats::dexp(x, rate = 1, log = TRUE),
        gamma = function(x) stats::dbeta(x, 2, 2, log = TRUE)
      ),
      settings = list(
        name = "Utility",
        mode = "fitting",
        estimate = "MLE",
        policy = "off"
      )
    )
  )

  for (scenario_name in base::names(scenarios)) {
    scenario <- scenarios[[scenario_name]]
    legacy_model <- legacy_tab_run(
      params = scenario$params,
      priors = scenario$priors,
      settings = scenario$settings
    )
    cpp_model <- cpp_tab_run(
      params = scenario$params,
      priors = scenario$priors,
      settings = scenario$settings
    )

    testthat::expect_equal(
      extract_cpp_metric(cpp_model),
      extract_legacy_metric(legacy_model),
      tolerance = 1e-8,
      info = scenario_name
    )

    testthat::expect_true("behave" %in% base::names(cpp_model$result))
    testthat::expect_equal(
      base::nrow(cpp_model$result$behave),
      base::nrow(tab_subject_one()) + 1L,
      info = scenario_name
    )
  }
})
