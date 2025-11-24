#' multiRL.metric
#'
#' @param output output
#' @param ... extra
#'
#' @returns multiRL.metric
#' 
process_5_metric <- function(
    output,
    ...
){
  
  extra <- list(...)
  
################################### [ACC] ######################################
  
  action      <- output@input@features@action
  simulation  <- output@result@simulation
  n_rows      <- output@input@n_rows
  ACC         <- sum(rowSums(action == simulation) == ncol(action)) / n_rows
  
  cue         <- output@behrule@cue
  rsp         <- output@behrule@rsp
  match       <- identical(cue, rsp)
  n_params    <- length(output@input@params@free)
  prob        <- output@result@prob
  LL          <- NA_real_
  AIC         <- NA_real_
  BIC         <- NA_real_
  
  priors      <- output@input@priors
  params      <- output@input@params@free
  post        <- .check_priors_params(priors = priors, params = params)
  LPr         <- NA_real_
  LPo         <- NA_real_
  
################################### [LL] #######################################
  
  if (match) {
    # 如果刺激和反应是一一对应, 才能计算LL
    P <- prob[cbind(seq_len(nrow(prob)), match(action, colnames(prob)))]
    logP <- log(P)
    LL <- sum(logP)
    AIC <- 2 * n_params - 2 * LL
    BIC <- n_params * log(n_rows) - 2 * LL

################################### [LP] #######################################    
    
    if (post) {
      # 如果在可计算LL的前提下, 还输入了对应的先验概率. 则计算LP
      LPr <- .calculate_log_prior(priors = priors, params = params)
      LPo <- LL + LPr
    }
  }

################################## [ABC] #######################################
 
  idinfo      <- output@input@features@idinfo
  latent      <- output@result@latent
  simulation  <- output@result@simulation
  behavior    <- as.data.frame(base::cbind(idinfo, latent, simulation))
  colnames(behavior) <- c("Subject", "Block", "Trial", "Latent", "Simulation")
  
  # 计算每个block中latent和simulation的选择比率
  ratio <- lapply(X = split(behavior, behavior[, "Block"]), FUN = function(x) {
    latent_prop <- .block_ratio(data = x, colname = "Latent", levels = cue)
    simul_prop  <- .block_ratio(data = x, colname = "Simulation", levels = rsp)
    list(Latent = latent_prop, Simulation = simul_prop)
  })
  
  onerow <- .for_abc(ratio)
  
  ABC <- list(ratio = ratio, onerow = onerow)
  
################################# [return] ##################################### 
  
  sumstat <- methods::new(
    Class = "multiRL.sumstat",
    ACC = ACC,
    LL = LL,
    AIC = AIC,
    BIC = BIC,
    LPr = LPr,
    LPo = LPo,
    ABC = ABC,
    extra = extra
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
