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
    
    raw     <- object@input@data
    value   <- object@result@value
    bias    <- object@result@bias
    prob    <- object@result@prob
    shown   <- object@result@shown
    count   <- object@result@count
    
    system  <- object@input@settings@system
    
    # 对小数点多的列, 保留两位小数
    value <- lapply(value, function(x) round(x = x, digits = 2))
    bias  <- round(bias, digits = 2)
    prob  <- round(prob, digits = 2)
    
    # 对不同类型的列增加前缀
    value <- lapply(value, function(x) {.prefix_colnames(as.data.frame(x), "V_")})
    bias  <- .prefix_colnames(as.data.frame(bias), "B_")
    prob  <- .prefix_colnames(as.data.frame(prob), "P_")
    shown <- .prefix_colnames(as.data.frame(object@result@shown), "S_")
    count <- .prefix_colnames(as.data.frame(object@result@count), "C_")
    
    others <- object@result@others

    behavior <- data.frame(
      Exploration = object@result@exploration,
      Latent      = object@result@latent,
      Reward      = object@result@reward,
      Utility     = object@result@utility,
      Simulation  = object@result@simulation,
      Position    = object@result@position
    )
    
    data <- cbind(raw, behavior, bias, shown, prob, count, others)
    process <- value
    names(process) <- system

    params  <- object@input@params
    metrics <- object@sumstat
    
    multiRL.summary <- methods::new(
      Class = "multiRL.summary",
      data = data,
      process = process,
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