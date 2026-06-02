summary.multiRLcpp_run_m <- function(object, ...) {
  fit <- object$fit

  if (base::nrow(fit) > 1L) {
    base::print(fit)
    return(base::invisible(object))
  }

  base::cat("Model Fit:\n")
  base::cat(
    "  Accuracy: ",
    base::round(fit$ACC * 100, 2),
    "%\n",
    sep = ""
  )
  base::cat(
    "  Log-Likelihood: ",
    base::round(fit$LogL, 2),
    "\n",
    sep = ""
  )

  if (!base::is.na(fit$LogPr)) {
    base::cat(
      "  Log-Prior Probability: ",
      base::round(fit$LogPr, 2),
      "\n",
      sep = ""
    )
  }

  if (!base::is.na(fit$LogPo)) {
    base::cat(
      "  Log-Posterior Probability: ",
      base::round(fit$LogPo, 2),
      "\n",
      sep = ""
    )
  }

  base::cat(
    "  AIC: ",
    base::round(fit$AIC, 2),
    "\n",
    sep = ""
  )
  base::cat(
    "  BIC: ",
    base::round(fit$BIC, 2),
    "\n",
    sep = ""
  )

  base::invisible(object)
}

summary.multiRLcpp_estimate_mle <- summary.multiRLcpp_run_m
summary.multiRLcpp_estimate_map <- summary.multiRLcpp_run_m
summary.multiRLcpp_run <- summary.multiRLcpp_run_m
