process_3_record <- function(
    input,
    behrule,
    ...
){
  
################################## [check] #####################################
  
  extra <- list(...)
  
  if (!is(input, "multiRL.input")) {
    stop("'input' must be an object of class 'multiRL.input'.")
  }
  
  if (!is(behrule, "multiRL.behrule")) {
    stop("'behrule' must be an object of class 'multiRL.behrule'.")
  }
  
############################### [null matrix] ##################################
  
  # 建立空表格记录价值更新
  nulldf <- matrix(
    data = NA_real_,
    nrow = input@n_trial * input@n_block,
    ncol = length(behrule@cue)
  )
  colnames(nulldf) <- behrule@cue
  
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
  
  exploration <- singledf
  latent      <- singledf
  reward      <- singledf
  utility     <- singledf
  simulation  <- singledf
  
  methods::setClass(
    Class = "multiRL.result",
    slots = list(
      value = "matrix",
      bias = "matrix",
      shown = "matrix",
      prob = "matrix",
      count = "matrix",
      
      exploration = "matrix",
      latent = "matrix",
      reward = "matrix",
      utility = "matrix",
      simulation = "matrix"
    )
  )
  
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
  
  methods::setClass(
    Class = "multiRL.record",
    slots = list(
      input = "multiRL.input",
      behrule = "multiRL.behrule",
      result = "multiRL.result",
      extra = "list"
    )
  )
  
  multiRL.record <- methods::new(
    Class = "multiRL.record",
    input = input,
    behrule = behrule,
    result = result,
    extra = extra
  )
  
  return(multiRL.record)
}
