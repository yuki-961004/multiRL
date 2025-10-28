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
  idinfo            <- model@input@features@idinfo
  state             <- model@input@features@state
  action            <- model@input@features@action
  latent            <- model@result@latent
  simulation        <- model@result@simulation

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
          array <- list(
            idinfo = idinfo, state = state,
            latent = latent, simulation = simulation
          )
          result <- list(data = data, params = params, array = array)
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
          array <- list(
            idinfo = idinfo, state = state,
            latent = latent, simulation = simulation
          )
          result <- list(
            array = array,
            params = params
          )
          return(result)
        },
      )
    }
  )
}
