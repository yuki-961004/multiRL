rpl_e <- function(
    result,
    option = list(plot = TRUE),
    ...
) {
  if (base::inherits(result, "multiRLcpp_rcv_d")) {
    return(.rpl_e_recovery(
      result = result,
      option = option,
      extra = list(...)
    ))
  }

  if (!base::inherits(result, "multiRLcpp_fit_p")) {
    base::stop(
      "rpl_e() requires a result from fit_p() or rcv_d().",
      call. = FALSE
    )
  }

  .rpl_e_fitting(
    result = result,
    option = option,
    extra = list(...)
  )
}

.rpl_e_fitting <- function(result, option, extra) {
  option <- .rpl_e_option(option)
  raw <- .rpl_e_raw_fits(result)
  replay_parts <- list()
  plot_parts <- list()
  human_seen <- FALSE

  for (name in base::names(raw)) {
    item <- raw[[name]]
    fit <- item$fit
    if (!"model" %in% base::names(fit)) {
      fit$model <- item$input$settings$name
    }
    if (!"model_id" %in% base::names(fit)) {
      fit$model_id <- name
    }

    cpp <- .rpl_e_call_cpp(item = item, fit = fit)
    replay_parts[[name]] <- cpp$replay

    plot_data <- cpp$plot_data
    if (human_seen) {
      plot_data <- plot_data[plot_data$source != "human", , drop = FALSE]
    }
    human_seen <- TRUE
    plot_parts[[name]] <- plot_data
  }

  replay <- .fit_p_rbind(replay_parts)
  plot_data <- .fit_p_rbind(plot_parts)

  out <- list(
    input = list(
      fit = .rpl_e_trim_fit(result),
      option = option,
      extra = extra
    ),
    replay = replay,
    plot_data = plot_data,
    plot = NULL,
    diagnostics = list(
      n_models = base::length(raw),
      n_subjects = base::length(base::unique(plot_data$subid)),
      policy = "on",
      replay_success = base::all(base::vapply(
        raw,
        function(x) TRUE,
        logical(1L)
      ))
    )
  )

  if (base::isTRUE(option$plot)) {
    out$plot <- .rpl_e_plot_fitting(plot_data)
    base::print(out$plot)
  }

  base::class(out) <- c("multiRLcpp_rpl_e", "multiRLcpp_run", "list")
  out
}

.rpl_e_recovery <- function(result, option, extra) {
  option <- .rpl_e_option(option)
  out <- list(
    input = list(
      recovery = result,
      option = option,
      extra = extra
    ),
    recovery = result$recovery,
    model_recovery = result$model_recovery,
    plot = NULL,
    diagnostics = list(
      n_generating = base::length(base::unique(
        result$model_recovery$generating_model
      )),
      n_candidates = base::length(base::unique(
        result$model_recovery$candidate_model
      ))
    )
  )

  if (base::isTRUE(option$plot)) {
    out$plot <- .rpl_e_plot_recovery(result)
    .rpl_e_print_plot(out$plot)
  }

  base::class(out) <- c("multiRLcpp_rpl_e", "multiRLcpp_run", "list")
  out
}

.rpl_e_option <- function(option) {
  if (base::is.null(option)) {
    option <- list()
  }
  utils::modifyList(
    x = list(plot = TRUE),
    val = option
  )
}

.rpl_e_raw_fits <- function(result) {
  if (!base::is.null(result$raw)) {
    return(result$raw)
  }
  name <- result$input$settings$name
  if (base::is.null(name)) {
    name <- "model"
  }
  out <- list(result)
  base::names(out) <- name
  out
}

.rpl_e_call_cpp <- function(item, fit) {
  input <- item$input
  params <- input$params
  priors <- input$priors
  settings <- input$settings
  features <- input$features

  .shell_rpl_e(
    object = features$object,
    reward = features$reward,
    action = features$action,
    block = features$block,
    trial = features$trial,
    idinfo = features$idinfo,
    exinfo = features$exinfo,
    cue = input$behrule$cue,
    rsp = input$behrule$rsp,
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
    fit = fit,
    parameter_names = base::names(params$free)
  )
}

.rpl_e_trim_fit <- function(result) {
  list(
    fit = result$fit,
    estimator = result$estimator,
    diagnostics = result$diagnostics
  )
}

.rpl_e_plot_fitting <- function(plot_data) {
  if (!base::requireNamespace("ggplot2", quietly = TRUE)) {
    base::stop(
      paste0(
        "rpl_e plotting requires ggplot2. Install ggplot2 or set ",
        "option = list(plot = FALSE)."
      ),
      call. = FALSE
    )
  }

  plot <- ggplot2::ggplot(
    plot_data,
    ggplot2::aes(
      x = block,
      y = ratio,
      color = model,
      group = model
    )
  ) +
    ggplot2::stat_summary(
      fun = "mean",
      geom = "line",
      linewidth = 1
    ) +
    ggplot2::stat_summary(
      fun.data = "mean_se",
      geom = "errorbar",
      width = 0.2
    ) +
    ggplot2::facet_wrap(stats::as.formula("~ action")) +
    ggplot2::scale_x_continuous(
      breaks = base::unique(plot_data$block)
    ) +
    ggplot2::scale_y_continuous(limits = c(0, 1)) +
    ggplot2::scale_color_manual(
      values = stats::setNames(
        .rpl_e_palette(base::length(base::unique(plot_data$model))),
        base::unique(plot_data$model)
      )
    ) +
    ggplot2::labs(
      x = "Block",
      y = "Action Ratio",
      color = "Model"
    )

  plot <- plot + .theme_apa()
  plot
}

.rpl_e_plot_recovery <- function(result) {
  if (!base::requireNamespace("ggplot2", quietly = TRUE)) {
    base::stop(
      paste0(
        "rpl_e plotting requires ggplot2. Install ggplot2 or set ",
        "option = list(plot = FALSE)."
      ),
      call. = FALSE
    )
  }

  ###########################
  # Preserve input model order
  ###########################
  # Extract model names from generating specs while preserving input order
  model_order <- base::vapply(result$input$generating, function(spec) {
    name <- spec$settings$name
    if (base::is.null(name)) {
      name <- spec$model
    }
    base::as.character(name[[1L]])
  }, FUN.VALUE = character(1L))
  # Convert model columns to factor with input order as levels
  result$model_recovery$generating_model <- base::factor(
    result$model_recovery$generating_model,
    levels = model_order
  )
  result$model_recovery$candidate_model <- base::factor(
    result$model_recovery$candidate_model,
    levels = model_order
  )
  result$recovery$generating_model <- base::factor(
    result$recovery$generating_model,
    levels = model_order
  )

  list(
    parameter = .rpl_e_plot_parameter_recovery(result),
    confusion = .rpl_e_plot_model_matrix(result$model_recovery, "confusion"),
    inversion = .rpl_e_plot_model_matrix(result$model_recovery, "inversion")
  )
}

.rpl_e_print_plot <- function(plot) {
  if (base::inherits(plot, "ggplot")) {
    base::print(plot)
    return(base::invisible(NULL))
  }
  if (base::is.list(plot)) {
    base::invisible(base::lapply(plot, .rpl_e_print_plot))
    return(base::invisible(NULL))
  }
  base::invisible(NULL)
}

.rpl_e_plot_parameter_recovery <- function(result) {
  recovery <- result$recovery
  recovery <- recovery[
    base::is.finite(recovery$true) &
      base::is.finite(recovery$recovered) &
      recovery$generating_model == recovery$candidate_model,
    ,
    drop = FALSE
  ]
  recovery$plot_true <- recovery$true
  recovery$plot_recovered <- recovery$recovered
  beta <- grepl("beta", recovery$parameter)
  positive <- recovery$plot_true > 0 & recovery$plot_recovered > 0
  recovery <- recovery[!beta | positive, , drop = FALSE]
  beta <- grepl("beta", recovery$parameter)
  recovery$plot_true[beta] <- base::log(recovery$plot_true[beta])
  recovery$plot_recovered[beta] <- base::log(
    recovery$plot_recovered[beta]
  )
  recovery$panel <- .rpl_e_parameter_panel(result, recovery)
  recovery$panel <- factor(recovery$panel, levels = base::unique(
    recovery$panel
  ))
  limits <- .rpl_e_parameter_limits(
    result = result,
    recovery = recovery
  )
  priors <- .rpl_e_parameter_prior_labels(
    result = result,
    recovery = recovery,
    limits = limits
  )
  limits$panel <- factor(limits$panel, levels = base::levels(
    recovery$panel
  ))
  priors$panel <- factor(priors$panel, levels = base::levels(
    recovery$panel
  ))

  models <- base::unique(recovery$generating_model)
  plots <- list()
  for (model in models) {
    local <- recovery[recovery$generating_model == model, , drop = FALSE]
    local_limits <- limits[limits$generating_model == model, , drop = FALSE]
    local_priors <- priors[priors$generating_model == model, , drop = FALSE]
    plot <- ggplot2::ggplot(
      local,
      ggplot2::aes(x = plot_true, y = plot_recovered)
    ) +
      ggplot2::geom_point(color = "#053562") +
      ggplot2::geom_abline(
        slope = 1,
        intercept = 0,
        linetype = "dashed",
        color = "#55c186"
      ) +
      ggplot2::facet_wrap(
        stats::as.formula("~ panel"),
        scales = "free",
        nrow = 1
      ) +
      ggplot2::geom_blank(
        data = local_limits,
        mapping = ggplot2::aes(x = limit, y = limit),
        inherit.aes = FALSE
      ) +
      ggplot2::geom_text(
        data = local_priors,
        mapping = ggplot2::aes(x = x, y = y, label = label),
        inherit.aes = FALSE,
        hjust = 1,
        vjust = 0,
        color = "#053562"
      ) +
      ggplot2::labs(x = "True", y = "Recovered")

  plot <- plot + .theme_apa()
    plot <- plot + ggplot2::theme(aspect.ratio = 1)
    plots[[model]] <- plot
  }
  plots
}

.rpl_e_parameter_panel <- function(result, recovery) {
  base::vapply(seq_len(base::nrow(recovery)), function(row) {
    model <- recovery$generating_model[[row]]
    parameter <- recovery$parameter[[row]]
    local <- recovery[
      recovery$generating_model == model &
        recovery$parameter == parameter,
      ,
      drop = FALSE
    ]
    correlation <- stats::cor(
      local$plot_true,
      local$plot_recovered,
      use = "complete.obs"
    )
    label <- paste0(
      model,
      ": ",
      parameter,
      " (r = ",
      base::formatC(correlation, digits = 2, format = "f"),
      ")"
    )
    label
  }, character(1L))
}

.rpl_e_plot_model_matrix <- function(model_recovery, type) {
  matrix <- .rpl_e_model_matrix(model_recovery, type)
  plot <- ggplot2::ggplot(
    matrix,
    ggplot2::aes(
      x = generating_model,
      y = candidate_model,
      fill = fill_value
    )
  ) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(
      ggplot2::aes(label = value),
      color = "white",
      fontface = "bold"
    ) +
    ggplot2::scale_fill_gradientn(
      colours = c("#e84a34", "#f0de36", "#55c186"),
      limits = c(0, 1),
      guide = "none"
    ) +
    ggplot2::scale_x_discrete(name = NULL) +
    ggplot2::scale_y_discrete(name = NULL) +
    ggplot2::labs(
      title = .rpl_e_matrix_title(type)
    )

  plot <- plot + .theme_apa()
  plot <- plot +
    ggplot2::theme(
      axis.title = ggplot2::element_blank(),
      axis.line = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank()
    )
  plot
}

.rpl_e_parameter_limits <- function(result, recovery) {
  panels <- base::unique(recovery$panel)
  rows <- list()
  cursor <- 1L
  for (panel in panels) {
    local <- recovery[recovery$panel == panel, , drop = FALSE]
    model <- local$generating_model[[1L]]
    parameter <- local$parameter[[1L]]
    lower <- .rpl_e_parameter_bound(result, model, parameter, "lower")
    upper <- .rpl_e_parameter_bound(result, model, parameter, "upper")
    beta <- grepl("beta", parameter)
    if (beta) {
      if (base::is.finite(lower) && lower > 0) {
        lower <- base::log(lower)
      } else {
        lower <- .rpl_e_log_magnitude_limit(
          local = local,
          side = "lower"
        )
      }
      if (base::is.finite(upper) && upper > 0) {
        upper <- .rpl_e_log_magnitude_limit(
          local = local,
          side = "upper"
        )
      } else {
        upper <- .rpl_e_log_magnitude_limit(
          local = local,
          side = "upper"
        )
      }
    }
    if (!base::is.finite(lower)) {
      lower <- base::min(local$plot_true, local$plot_recovered)
    }
    if (!base::is.finite(upper)) {
      upper <- base::max(local$plot_true, local$plot_recovered)
    }
    rows[[cursor]] <- data.frame(
      generating_model = model,
      panel = panel,
      limit = lower,
      stringsAsFactors = FALSE
    )
    rows[[cursor + 1L]] <- data.frame(
      generating_model = model,
      panel = panel,
      limit = upper,
      stringsAsFactors = FALSE
    )
    cursor <- cursor + 2L
  }
  out <- base::do.call(rbind, rows)
  out$panel <- factor(out$panel, levels = base::levels(recovery$panel))
  out
}

.rpl_e_parameter_prior_labels <- function(result, recovery, limits) {
  panels <- base::unique(recovery$panel)
  rows <- list()
  cursor <- 1L
  for (panel in panels) {
    local <- recovery[recovery$panel == panel, , drop = FALSE]
    local_limits <- limits[limits$panel == panel, , drop = FALSE]
    model <- local$generating_model[[1L]]
    parameter <- local$parameter[[1L]]
    label <- .rpl_e_parameter_prior(result, model, parameter)
    if (!base::nzchar(label)) {
      next
    }
    rows[[cursor]] <- data.frame(
      generating_model = model,
      panel = panel,
      x = base::max(local_limits$limit),
      y = base::min(local_limits$limit),
      label = label,
      stringsAsFactors = FALSE
    )
    cursor <- cursor + 1L
  }
  if (base::length(rows) == 0L) {
    return(data.frame(
      generating_model = character(),
      panel = character(),
      x = numeric(),
      y = numeric(),
      label = character(),
      stringsAsFactors = FALSE
    ))
  }
  base::do.call(rbind, rows)
}

.rpl_e_parameter_prior <- function(result, model, parameter) {
  specs <- result$input$generating
  for (spec in specs) {
    name <- spec$settings$name
    if (base::is.null(name)) {
      name <- spec$model
    }
    if (!base::identical(base::as.character(name[[1L]]), model)) {
      next
    }
    prior <- .rpl_e_prior_entry(spec$priors, parameter)
    if (base::is.null(prior)) {
      return("")
    }
    return(.rpl_e_format_prior(parameter, prior))
  }
  ""
}

.rpl_e_prior_entry <- function(priors, parameter) {
  if (base::is.null(priors)) {
    return(NULL)
  }
  names <- base::names(priors)
  if (base::is.null(names) || !parameter %in% names) {
    return(NULL)
  }
  priors[[parameter]]
}

.rpl_e_format_prior <- function(parameter, prior) {
  if (base::is.null(prior$type)) {
    return("")
  }
  type <- base::tolower(base::as.character(prior$type[[1L]]))
  label <- switch(
    type,
    exponential = "exp",
    normal = "norm",
    uniform = "unif",
    lognormal = "lnorm",
    type
  )
  values <- .rpl_e_prior_values(prior)
  if (base::length(values) == 0L) {
    return(label)
  }
  paste0(label, "(", paste(values, collapse = ", "), ")")
}

.rpl_e_prior_values <- function(prior) {
  names <- base::setdiff(base::names(prior), "type")
  if (base::length(names) == 0L) {
    return(character())
  }
  values <- character()
  for (name in names) {
    value <- prior[[name]][[1L]]
    numeric <- base::suppressWarnings(base::as.numeric(value))
    if (base::length(numeric) == 1L && !base::is.na(numeric)) {
      if (!base::is.finite(numeric)) {
        next
      }
    }
    text <- base::as.character(value)
    if (!base::is.na(text) && base::nzchar(text)) {
      values <- c(values, text)
    }
  }
  values
}

.rpl_e_log_magnitude_limit <- function(local, side) {
  values <- base::c(local$true, local$recovered)
  values <- values[base::is.finite(values) & values > 0]
  if (base::length(values) == 0L) {
    return(NA_real_)
  }
  if (side == "lower") {
    exponent <- base::floor(base::log10(base::min(values)))
  } else {
    exponent <- base::ceiling(base::log10(base::max(values)))
  }
  base::log(10^exponent)
}

.rpl_e_parameter_bound <- function(result, model, parameter, side) {
  specs <- result$input$generating
  for (spec in specs) {
    name <- spec$settings$name
    if (base::is.null(name)) {
      name <- spec$model
    }
    if (!base::identical(base::as.character(name[[1L]]), model)) {
      next
    }
    bound <- spec[[side]]
    value <- .rpl_e_bound_value(bound, parameter)
    if (base::is.finite(value)) {
      return(value)
    }
  }
  NA_real_
}

.rpl_e_bound_value <- function(bound, parameter) {
  if (base::is.null(bound)) {
    return(NA_real_)
  }
  names <- base::names(bound)
  if (base::is.null(names) || !parameter %in% names) {
    return(NA_real_)
  }
  value <- bound[parameter]
  if (base::is.list(value)) {
    value <- value[[1L]]
  }
  base::suppressWarnings(base::as.numeric(value[[1L]]))
}

.rpl_e_matrix_title <- function(type) {
  if (type == "confusion") {
    return("Confusion Matrix")
  }
  "Inversion Matrix"
}

.rpl_e_model_matrix <- function(model_recovery, type) {
  generating <- base::unique(model_recovery$generating_model)
  candidate <- base::unique(model_recovery$candidate_model)
  grid <- base::expand.grid(
    generating_model = generating,
    candidate_model = candidate,
    stringsAsFactors = FALSE
  )
  selected <- model_recovery[model_recovery$selected, , drop = FALSE]
  if (base::nrow(selected) == 0L) {
    count <- data.frame(
      generating_model = character(),
      candidate_model = character(),
      count = integer(),
      stringsAsFactors = FALSE
    )
  } else {
    count <- stats::aggregate(
      x = list(count = selected$selected),
      by = list(
        generating_model = selected$generating_model,
        candidate_model = selected$candidate_model
      ),
      FUN = base::length
    )
  }
  out <- base::merge(
    x = grid,
    y = count,
    by = c("generating_model", "candidate_model"),
    all.x = TRUE,
    sort = FALSE
  )
  out$count[base::is.na(out$count)] <- 0L

  if (type == "confusion") {
    denominator <- .rpl_e_matrix_denominator(
      selected = selected,
      group = "generating_model"
    )
    out <- base::merge(
      x = out,
      y = denominator,
      by = "generating_model",
      all.x = TRUE,
      sort = FALSE
    )
  } else {
    denominator <- .rpl_e_matrix_denominator(
      selected = selected,
      group = "candidate_model"
    )
    out <- base::merge(
      x = out,
      y = denominator,
      by = "candidate_model",
      all.x = TRUE,
      sort = FALSE
    )
  }
  out$value <- 0
  valid <- base::is.finite(out$total) & out$total > 0
  out$value[valid] <- out$count[valid] / out$total[valid]
  out$value <- base::round(out$value, 2)
  out$fill_value <- 1 - out$value
  diagonal <- out$generating_model == out$candidate_model
  out$fill_value[diagonal] <- out$value[diagonal]
  out[
    ,
    c("generating_model", "candidate_model", "value", "fill_value"),
    drop = FALSE
  ]
}

.rpl_e_matrix_denominator <- function(selected, group) {
  if (base::nrow(selected) == 0L) {
    out <- data.frame(total = integer(), stringsAsFactors = FALSE)
    out[[group]] <- character()
    return(out[, c(group, "total"), drop = FALSE])
  }
  stats::aggregate(
    x = list(total = selected$selected),
    by = stats::setNames(list(selected[[group]]), group),
    FUN = base::length
  )
}

.rpl_e_palette <- function(n) {
  base_colors <- c(
    "grey",
    "#053562",
    "#55c186",
    "#f0de36",
    "#f79d1e",
    "#e84a34",
    "#8b2f97"
  )
  if (n <= base::length(base_colors)) {
    return(base_colors[base::seq_len(n)])
  }
  dynamic_cols <- grDevices::colorRampPalette(base_colors[-1])(n - 1L)
  c(base_colors[1], dynamic_cols)
}

.theme_apa <- function(base_size = 12, base_family = "", box = FALSE) {
  adapted_theme <- ggplot2::theme_bw(base_size, base_family) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(
        size   = ggplot2::rel(1.1),
        margin = ggplot2::margin(0, 0, ggplot2::rel(14), 0),
        hjust  = 0.5
      ),
      plot.subtitle = ggplot2::element_text(
        size   = ggplot2::rel(0.8),
        margin = ggplot2::margin(
          ggplot2::rel(-7), 0, ggplot2::rel(14), 0
        ),
        hjust  = 0.5
      ),
      axis.title.x = ggplot2::element_text(
        size       = ggplot2::rel(1),
        lineheight = ggplot2::rel(1.1),
        margin     = ggplot2::margin(ggplot2::rel(12), 0, 0, 0)
      ),
      axis.title.x.top = ggplot2::element_text(
        size       = ggplot2::rel(1),
        lineheight = ggplot2::rel(1.1),
        margin     = ggplot2::margin(0, 0, ggplot2::rel(12), 0)
      ),
      axis.title.y = ggplot2::element_text(
        size       = ggplot2::rel(1),
        lineheight = ggplot2::rel(1.1),
        margin     = ggplot2::margin(0, ggplot2::rel(12), 0, 0)
      ),
      axis.title.y.right = ggplot2::element_text(
        size       = ggplot2::rel(1),
        lineheight = ggplot2::rel(1.1),
        margin     = ggplot2::margin(0, 0, 0, ggplot2::rel(12))
      ),
      axis.ticks.length = ggplot2::unit(
        ggplot2::rel(6), "points"
      ),
      axis.text = ggplot2::element_text(
        size = ggplot2::rel(0.9)
      ),
      axis.text.x = ggplot2::element_text(
        size   = ggplot2::rel(1),
        margin = ggplot2::margin(ggplot2::rel(6), 0, 0, 0)
      ),
      axis.text.y = ggplot2::element_text(
        size   = ggplot2::rel(1),
        margin = ggplot2::margin(0, ggplot2::rel(6), 0, 0)
      ),
      axis.text.y.right = ggplot2::element_text(
        size   = ggplot2::rel(1),
        margin = ggplot2::margin(0, 0, 0, ggplot2::rel(6))
      ),
      axis.line = ggplot2::element_line(),
      legend.title = ggplot2::element_text(),
      legend.key = ggplot2::element_rect(
        fill  = NA,
        color = NA
      ),
      legend.key.width = ggplot2::unit(
        ggplot2::rel(20), "points"
      ),
      legend.key.height = ggplot2::unit(
        ggplot2::rel(20), "points"
      ),
      legend.margin = ggplot2::margin(
        t    = ggplot2::rel(16),
        r    = ggplot2::rel(16),
        b    = ggplot2::rel(16),
        l    = ggplot2::rel(16),
        unit = "points"
      ),
      panel.spacing = ggplot2::unit(
        ggplot2::rel(14), "points"
      ),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.minor.x = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_blank(),
      panel.grid.minor.y = ggplot2::element_blank(),
      strip.background = ggplot2::element_rect(
        fill  = NA,
        color = NA
      ),
      strip.text.x = ggplot2::element_text(
        size   = ggplot2::rel(1.2),
        margin = ggplot2::margin(0, 0, ggplot2::rel(10), 0)
      ),
      strip.text.y = ggplot2::element_text(
        size   = ggplot2::rel(1.2),
        margin = ggplot2::margin(0, 0, 0, ggplot2::rel(10))
      )
    )
  if (box) {
    adapted_theme <- adapted_theme +
      ggplot2::theme(
        panel.border = ggplot2::element_rect(color = "black")
      )
  } else {
    adapted_theme <- adapted_theme +
      ggplot2::theme(
        panel.border = ggplot2::element_blank()
      )
  }
  adapted_theme
}
