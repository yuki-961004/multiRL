#' plot.multiRL.replay
#'
#' @param x multiRL.replay
#' @param y NULL
#' @param model 
#'  The name of model that you want to plot
#' @param param 
#'  The name of parameter that you want to plot
#' @param ... extra
#'
#' @returns List
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
