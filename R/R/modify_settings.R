.modify_settings <- function(settings) {
  default_settings <- list(
    name = "unknown",
    mode = "fitting",
    estimate = "MLE",
    policy = "on",
    system = "RL"
  )

  utils::modifyList(default_settings, settings)
}
