process_4_loop <- function(
    record,
    rate_func,
    util_func,
    bias_func,
    prob_func,
    ...
){
################################## [check] #####################################
  
  # 额外信息
  extra <- list(...)
  
  if (!is(record, "multiRL.record")) {
    stop("'record' must be an object of class 'multiRL.record'.")
  }
  
  state <- record@input@features@state
  
  cue <- record@behrules@cue
  rsp <- record@behrules@rsp
  
  value <- record@output@value
  bias <- record@output@bias
  utility <- record@output@utility
  shown <- record@output@shown
  prob <- record@output@prob
  
  latent <- record@output@latent
  reward <- record@output@reward
  simulation <- record@output@simulation
  
  set.seed(123)
  
  for (i in 1:(multiRL.record@input@n_rows - 1)) {
    
    # 只有第一次需要获取初始值
    if (i == 1) {value[i, ] <- initial_value} 
    
    # 偏差值func_pi
    bias[i, ] <- 0
    # 选项是否出现
    shown[i, ] <- stats::setNames(
      object = as.numeric(cue %in% state[i, , ]),
      nm = cue
    )
    # 主观价值func_gamma
    utility[i, ] <- util_func(x = value[i, ], gamma = 2)
    utility[i, ] <- utility[i, ] * shown[i, ]
    # 价值转概率func_beta
    prob[i, ] <- prob_func(x = utility[i, ], beta = 1)
    # 根据概率, 进行选择内在选择
    latent[i, ] <- sample(x = colnames(prob), prob = prob[i, ], size = 1)
    # 内在选择所在行
    row_index <- which(state[i, , ] == latent[i, ])
    # 外在选择所在列
    col_index <- which(state[i, row_index, ] %in% rsp)
    # 根据内在选择[行, 列], 定位外在选择
    simulation[i, ] <- state[i, row_index, col_index]
    # 根据内在选择定位奖励
    reward[i, ] <- state[i, row_index, dim(state)[3]]
    # 传递到下一行
    value[i + 1, ] <- value[i, ]
    # 基于PE价值更新
    value[i + 1, latent[i, ]] <- rate_func(
      qvalue = as.numeric(value[i, latent[i, ]]),
      reward = as.numeric(reward[i, ]), 
      alpha = 0.5
    )
  }
  
  record@output@value <- value
  record@output@bias <- bias
  record@output@utility <- utility
  record@output@shown <- shown
  record@output@prob <- prob
  
  record@output@latent <- latent
  record@output@reward <- reward
  record@output@simulation <- simulation
  
  return(record)
}
