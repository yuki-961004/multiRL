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
  
  # basic
  data              <- multiRL.summary@data
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
      result <- list(data = data, input_params = params)
      return(result)
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
