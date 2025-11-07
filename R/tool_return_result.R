#' .return_result
#'
#' @param model multiRL.model
#'
#' @returns result
#'
#' @noRd
#' 
.return_result <- function(model) {
  
  multiRL.summary   <- suppressMessages(multiRL::summary(model))
  mode              <- model@input@settings@mode
  estimate          <- model@input@settings@estimate
  
  # 将模拟数据集变成和原始数据集一样的列
  data              <- multiRL.summary@data
  colnames          <- colnames(model@input@data)
  data <- data[, colnames]
  data[, model@input@colnames@action] <- as.vector(model@result@simulation)
  
  params            <- model@input@params@free

  # for MLE
  LL                <- model@sumstat@LL
  
  # for MAP 
  LPo               <- model@sumstat@LPo
  
  # for ABC
  sumstat           <- model@sumstat@ABC$onerow
  
  # for RNN
  matrix            <- .df2matrix(df = data)
  
  switch(
    EXPR = mode,
    "simulating" = {
      switch(
        EXPR = estimate,
        "MLE" = {
          result <- list(data = data, params = params, LL = LL)
          return(result)
        },
        "MAP" = {
          result <- list(data = data, params = params, LPo = LPo)
          return(result)
        },
        "ABC" = {
          result <- list(data = data, params = params, sumstat = sumstat)
          return(result)
        },
        "RNN" = {
          result <- list(data = data, params = params, matrix = matrix)
          return(result)
        },
      )
    },
    "fitting" = {
      switch(
        EXPR = estimate,
        "MLE" = {
          return(LL)
        },
        "MAP" = {
          return(LPo)
        },
        "ABC" = {
          return(sumstat)
        },
        "RNN" = {
          return(matrix)
        },
      )
    }
  )
}
