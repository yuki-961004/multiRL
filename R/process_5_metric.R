process_5_metric <- function(
    output,
    ...
){
################################## [check] #####################################
  
  extra <- list(...)
  
  if (!is(output, "multiRL.output")) {
    stop("'output' must be an object of class 'multiRL.output'.")
  }
  
  action      <- output@input@features@action
  simulation  <- output@result@simulation
  n_rows      <- output@input@n_rows
  ACC         <- sum(rowSums(action == simulation) == ncol(action)) / n_rows
  
  match       <- identical(output@behrule@cue, output@behrule@rsp)
  prob        <- output@result@prob
  n_params    <- length(output@input@params@free)
  
  if (match) {
    prob <- prob[cbind(seq_len(nrow(prob)), match(action, colnames(prob)))]
    
    # 计算 logP
    logP <- base::log(prob)
    
    LL <- sum(logP)
    AIC <- round(2 * n_params - 2 * LL, digits = 2)
    BIC <- round(n_params * log(n_rows) - 2 * LL, digits = 2)
  }
  else {
    LL <- NA_real_
    AIC <- NA_real_
    BIC <- NA_real_
  }
  
  methods::setClass(
    Class = "multiRL.sumstat",
    slots = list(
      ACC = "numeric",
      LL = "numeric",
      AIC = "numeric",
      BIC = "numeric",
      extra = "list"
    )
  )
  
  sumstat <- methods::new(
    Class = "multiRL.sumstat",
    ACC = ACC,
    LL = LL,
    AIC = AIC,
    BIC = BIC,
    extra = extra
  )
  
  methods::setClass(
    Class = "multiRL.metric",
    slots = list(
      input = "multiRL.input",
      behrule = "multiRL.behrule",
      result = "multiRL.result",
      sumstat = "multiRL.sumstat",
      extra = "list"
    )
  )
  
  metric <- methods::new(
    Class = "multiRL.metric",
    input = output@input,
    behrule = output@behrule,
    result = output@result,
    sumstat = sumstat,
    extra = output@extra
  )
  
  return(metric)
}
