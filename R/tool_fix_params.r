
.fix_params <- function(params_df, param_names, lower, upper, dash) {
  for (i in seq_along(param_names)) {
    p_name <- param_names[i]
    lower[i] <- lower[i] + dash
    upper[i] <- upper[i] - dash
    if (p_name %in% colnames(params_df)) {
      params_df[, p_name] <- pmax(params_df[, p_name], lower[i])
      params_df[, p_name] <- pmin(params_df[, p_name], upper[i])
    }
  }
  return(params_df)
}