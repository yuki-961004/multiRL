process_3_record <- function(
  input,
  behrules,
  ...
){
################################## [check] #####################################
  
  # 额外信息
  extra <- list(...)
  
  if (!is(input, "multiRL.input")) {
    stop("'input' must be an object of class 'multiRL.input'.")
  }
  
  if (!is(behrules, "multiRL.behrules")) {
    stop("'behrules' must be an object of class 'multiRL.behrules'.")
  }
  
############################### [null matrix] ##################################
  
  # 建立空表格记录价值更新
  nulldf <- matrix(
    data = NA_real_,
    nrow = input@n_trial * input@n_block,
    ncol = length(multiRL.behrules@cue)
  )
  colnames(nulldf) <- multiRL.behrules@cue
  
  singledf <- matrix(
    data = NA_real_,
    nrow = input@n_trial * input@n_block,
    ncol = 1
  )
  
  value <- nulldf
  bias <- nulldf
  utility <- nulldf
  shown <- nulldf
  prob <- nulldf
  
  latent <- singledf
  reward <- singledf
  simulation <- singledf
  
  methods::setClass(
    Class = "multiRL.output",
    slots = list(
      value = "matrix",
      bias = "matrix",
      utility = "matrix",
      shown = "matrix",
      prob = "matrix",
      latent = "matrix",
      reward = "matrix",
      simulation = "matrix"
    )
  )
  
  multiRL.output <- methods::new(
    Class = "multiRL.output",
    value = value,
    bias = bias,
    utility = utility,
    shown = shown,
    prob = prob,
    latent = latent,
    reward = reward,
    simulation = simulation
  )
  
################################## [record] ####################################
  
  methods::setClass(
    Class = "multiRL.record",
    slots = list(
      input = "multiRL.input",
      behrules = "multiRL.behrules",
      output = "multiRL.output",
      extra = "list"
    )
  )
  
  multiRL.record <- methods::new(
    Class = "multiRL.record",
    input = input,
    behrules = behrules,
    output = multiRL.output,
    extra = extra
  )
  
  return(multiRL.record)
}
