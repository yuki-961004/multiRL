#' summary
#'
#' @param object multiRL.model. 
#' @param ... ...
#'
#' @returns multiRL.summary
#' 
methods::setMethod(
  f = "summary",
  signature = methods::signature(object = "multiRL.model"),
  definition = function(object, ...) {

################################## [return] ####################################
    
    raw <- object@input@data
    
    object@result@value <- round(x = object@result@value, digits = 2)
    object@result@bias  <- round(x = object@result@bias, digits = 2)
    object@result@prob  <- round(x = object@result@prob, digits = 2)
    
    vlaue <- .prefix_colnames(as.data.frame(object@result@value), "V_")
    bias  <- .prefix_colnames(as.data.frame(object@result@bias), "B_")
    shown <- .prefix_colnames(as.data.frame(object@result@shown), "S_")
    prob  <- .prefix_colnames(as.data.frame(object@result@prob), "P_")
    count <- .prefix_colnames(as.data.frame(object@result@count), "C_")
    
    behavior <- data.frame(
      Exploration = object@result@exploration,
      Latent      = object@result@latent,
      Reward      = object@result@reward,
      Utility     = object@result@utility,
      Simulation  = object@result@simulation
    )
    
    data    <- cbind(raw, vlaue, bias, shown, prob, count, behavior)
    params  <- object@input@params
    metrics <- object@sumstat
    
    multiRL.summary <- methods::new(
      Class = "multiRL.summary",
      data = data,
      params = params, 
      metrics = metrics
    )
    
################################# [message] ####################################
    
    ACC   <- round(multiRL.summary@metrics@ACC * 100, 2)
    LL    <- round(multiRL.summary@metrics@LL, 2)
    AIC   <- round(multiRL.summary@metrics@AIC, 2)
    BIC   <- round(multiRL.summary@metrics@BIC, 2)
    LPr   <- round(multiRL.summary@metrics@LPr, 2)
    LPo   <- round(multiRL.summary@metrics@LPo, 2)
    
    message(
      "Model Fit:\n",
      # Indent model fit metrics
      "  ", "Accuracy: ", ACC, "%\n",
      "  ", "Log-Likelihood: ", LL, "\n",
      "  ", "Log-Prior Probability: ", LPr, "\n",
      "  ", "Log-Posterior Probability: ", LPo, "\n",
      "  ", "AIC: ", AIC, "\n",
      "  ", "BIC: ", BIC,"\n"
    )
    
    return(multiRL.summary)
  }
)