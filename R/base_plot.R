#' plot.multiRL.replay
#'
#' @param x multiRL.replay
#' @param y NULL
#' @param model model
#' @param param param
#' @param ... extra
#'
#' @returns plot
#' 
plot.multiRL.replay <- function(
    x, y = NULL,
    model = NULL, param = NULL,
    ...
) {
  
  if (identical(c("simulate", "recovery"), names(x))) {
    .plot_recovery(x = x, model = model, param = param)
  } else {
    .plot_fitting(x = x)
  }
  
}
