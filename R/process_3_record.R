#' multiRL.record
#'
#' @param input input
#' @param behrule behrule
#' @param ... extra
#'
#' @returns multiRL.record
#' 
process_3_record <- function(
    input,
    behrule,
    ...
){
  
################################## [check] #####################################
  
  extra <- list(...)
  
############################### [null matrix] ##################################
  
  # 生成行数等量的多列表格
  nulldf <- matrix(
    data = NA_real_,
    nrow = input@n_trial * input@n_block,
    ncol = length(behrule@cue)
  )
  # 列数表示需要更新的价值, 即潜在规则
  colnames(nulldf) <- behrule@cue
  
  # 生成行数等量的单列表格
  singledf <- matrix(
    data = NA_real_,
    nrow = input@n_trial * input@n_block,
    ncol = 1
  )
  
  value       <- nulldf
  bias        <- nulldf
  shown       <- nulldf
  prob        <- nulldf
  count       <- nulldf
  
  exploration <- matrix(as.numeric(singledf), nrow = nrow(singledf), ncol = 1)
  latent      <- matrix(as.character(singledf), nrow = nrow(singledf), ncol = 1)
  reward      <- matrix(as.numeric(singledf), nrow = nrow(singledf), ncol = 1)
  utility     <- matrix(as.numeric(singledf), nrow = nrow(singledf), ncol = 1)
  simulation  <- matrix(as.character(singledf), nrow = nrow(singledf), ncol = 1)
  
  result <- methods::new(
    Class = "multiRL.result",
    value = value,
    bias = bias,
    shown = shown,
    prob = prob,
    count = count,
    
    exploration = exploration,
    latent = latent,
    reward = reward,
    utility = utility,
    simulation = simulation
  )
  
################################## [record] ####################################

  multiRL.record <- methods::new(
    Class = "multiRL.record",
    input = input,
    behrule = behrule,
    result = result,
    extra = extra
  )
  
  return(multiRL.record)
}
