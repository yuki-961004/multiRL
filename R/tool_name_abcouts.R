.name_abcouts <- function(summary, metric = "mode", param_names) {
  
  # 1. Extract matrix from list (happens when abc uses neuralnet or loclinear)
  if (is.list(summary) && !is.data.frame(summary)) {
    if (!is.null(summary$adj)) {
      summary <- summary$adj
    } else if (!is.null(summary$unadj)) {
      summary <- summary$unadj
    }
  }
  
  if (inherits(summary, "table")) {
    summary <- unclass(summary)
  }
  
  # 2. Regex patterns to identify each row in the abc summary matrix
  patterns <- c(
    "min"    = "min",
    "025"    = "2\\.5",
    "median" = "median",
    "mean"   = "mean",
    "mode"   = "mode",
    "975"    = "97\\.5",
    "max"    = "max"
  )
  
  stats_names <- tolower(rownames(summary))
  
  # Find row indices for each statistic
  row_indices <- list()
  for (p in names(patterns)) {
    idx <- grep(patterns[[p]], stats_names)
    row_indices[[p]] <- if (length(idx) > 0) idx[1] else NA
  }

  metric <- tolower(metric)
  if (!(metric %in% names(patterns))) {
    metric <- "mode" # Fallback
  }
  
  # Determine which row is the requested main metric
  base_idx <- row_indices[[metric]]
  if (is.na(base_idx)) {
    base_idx <- row_indices[["mean"]] 
  }
  
  out <- c()
  out_names <- c()
  
  for (i in seq_along(param_names)) {
    p_name <- param_names[i]
    
    # 1. The requested metric gets the plain parameter name
    if (!is.na(base_idx)) {
      out <- c(out, summary[base_idx, i])
      out_names <- c(out_names, p_name)
    }
    
    # 2. Other metrics get their corresponding suffixes
    for (p in names(patterns)) {
      idx <- row_indices[[p]]
      # Only append if it's found AND it's not the base metric
      if (!is.na(idx) && (is.na(base_idx) || idx != base_idx)) {
        out <- c(out, summary[idx, i])
        out_names <- c(out_names, paste0(p_name, "_", p))
      }
    }
  }
  
  names(out) <- out_names
  return(out)
}
