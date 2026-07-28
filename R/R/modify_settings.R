.modify_settings <- function(settings) {
  default_settings <- list(
    name = "unknown",
    mode = "fitting",
    estimate = "MLE",
    generate = FALSE,
    system = "RL"
  )

  utils::modifyList(default_settings, settings)
}
