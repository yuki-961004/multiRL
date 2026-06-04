library(multiRLcpp)

base_args <- list(
  data = binaryRL::Mason_2024_G2,
  id = c(1:3),
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
  settings = list(policy = "off"),
  lower = c(0, 0),
  upper = c(1, 5)
)

run_scope <- function(scope) {
  result <- do.call(
    what = fit_p,
    args = c(
      base_args,
      list(
        estimator = "abc",
        control = list(
          samples = 20,
          tol = 0.2,
          scope = scope,
          seed = 123,
          threads = 1
        )
      )
    )
  )

  stopifnot(base::is.list(result))
  stopifnot(
    base::all(c("input", "fit", "estimator", "diagnostics") %in%
      base::names(result))
  )
  stopifnot(base::identical(result$estimator$scope, scope))

  print(scope)
  print(result$fit)
  print(result$diagnostics$scope)
  invisible(result)
}

res_individual <- run_scope("individual")
res_shared <- run_scope("shared")
res_universal <- run_scope("universal")
