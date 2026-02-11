#' plot.multiRL.replay
#'
#' @param x multiRL.replay
#' @param y NULL
#' @param model 
#'  The name of model that you want to plot
#' @param param 
#'  The name of parameter that you want to plot
#' @param metric 
#'  The metric for identifying the optimal model defaults to BIC; if the LogL 
#'    cannot be calculated, ACC is used instead.
#' @param ... extra
#'
#' @returns An S3 object of class \code{ggplot2}
#' 
plot.multiRL.replay <- function(
    x, y = NULL,
    model = NULL, param = NULL, metric = "BIC",
    ...
) {
  
  if (identical(c("simulate", "recovery"), names(x))) {
    .plot_recovery(x = x, model = model, param = param, metric = metric)
  } else {
    .plot_fitting(x = x)
  }
  
}
