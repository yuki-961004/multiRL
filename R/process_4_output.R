#' multiRL.output
#'
#' @param record record
#' @param ... extra
#'
#' @returns multiRL.output
#' 
process_4_output <- function(
    record,
    ...
){
  
  extra <- list(...)
  
################################## [load] ######################################
  
  policy      <- record@input@settings@policy
  
  idinfo      <- record@input@features@idinfo
  state       <- record@input@features@state
  action      <- record@input@features@action
  exinfo      <- record@input@features@exinfo
  
  params      <- record@input@params
  
  rate_func   <- record@input@funcs@rate_func
  prob_func   <- record@input@funcs@prob_func
  util_func   <- record@input@funcs@util_func
  bias_func   <- record@input@funcs@bias_func
  expl_func   <- record@input@funcs@expl_func
  
  cue         <- record@behrule@cue
  rsp         <- record@behrule@rsp
  
  value       <- record@result@value
  bias        <- record@result@bias
  utility     <- record@result@utility
  shown       <- record@result@shown
  prob        <- record@result@prob
  count       <- record@result@count
  
  exploration <- record@result@exploration
  latent      <- record@result@latent
  reward      <- record@result@reward
  simulation  <- record@result@simulation
  
############################# [initial value] ##################################
  
  Q1          <- .get_param(params, "Q1")
  value[1, ]  <- ifelse(is.na(Q1), yes = 0, no = Q1)
  count[1, ]  <- 0
  value       <- rbind(value, rep(NA, ncol(value)))
  count       <- rbind(count, rep(NA, ncol(count)))
  n_rows      <- record@input@n_rows
  
############################# [action select] ##################################
  
  set.seed(123)
  
  for (i in 1:n_rows) {
    
    # bias function: 每个刺激上的偏见
    bias[i, ] <- bias_func(
      count = count[i, ], 
      params = params,
      idinfo = idinfo[i, ],
      exinfo = exinfo[i, ]
    )
    # 记录每个刺激是否出现
    shown[i, ] <- stats::setNames(
      object = ifelse(
        test = cue %in% state[i, , ],
        yes = 1,
        no = NA
      ),
      nm = cue
    )
    # exploration function: 此次是否进行探索
    exploration[i, ] <- expl_func(
      rownum = i,
      params = params,
      idinfo = idinfo[i, ],
      exinfo = exinfo[i, ]
    )
    # probability function: 选择每个选项的概率 
    prob[i, ] <- prob_func(
      qvalue = (value[i, ] + bias[i, ]) * shown[i, ], 
      explor = exploration[i, ],
      params = params,
      idinfo = idinfo[i, ],
      exinfo = exinfo[i, ]
    )
    
    switch(
      EXPR = policy,
      # on-policy: 基于机器人估计的概率进行选择
      "on" = {
        latent[i, ] <- sample(
          x = colnames(prob)[!is.na(prob[i, ])],
          prob = prob[i, which(!is.na(prob[i, ]))] / 
            sum(prob[i, which(!is.na(prob[i, ]))]),
          size = 1
        )
        row_index <- which(state[i, , ] == latent[i, ])
        col_index <- which(state[i, row_index, ] %in% rsp)
        simulation[i, ] <- state[i, row_index, col_index]
      },
      # on-policy: 复制人类的真实选择
      "off" = {
        latent[i, ] <- action[i, ]
        row_index <- which(state[i, , ] == latent[i, ])
        col_index <- which(state[i, row_index, ] %in% rsp)
        simulation[i, ] <- action[i, ]
      }
    )
    
############################## [value update] ##################################
    
    # 读取此时的奖励
    reward[i, ] <- state[i, row_index, dim(state)[3]]
    # utility function: 将实际奖励转化为主管价值
    utility[i, ] <- util_func(
      reward = as.numeric(reward[i, ]), 
      params = params,
      idinfo = idinfo[i, ],
      exinfo = exinfo[i, ]
    )
    
    # 继承上一列的所有值
    value[i + 1, ] <- value[i, ]
    # 记录此时被选选项的值
    Qi <- value[i, latent[i, ]]
    
    if (is.na(Q1) & Qi == 0) {
      # 如果是第一次选, 则直接记录价值 (等同于学习率100%的价值更新)
      value[i + 1, latent[i, ]] <- utility[i, ]
    } else {
      # learning rate function: 如果不是第一次选, 则按照学习率方程更新
      value[i + 1, latent[i, ]] <- rate_func(
        qvalue = as.numeric(value[i, latent[i, ]]),
        reward = as.numeric(utility[i, ]), 
        params = params,
        idinfo = idinfo[i, ],
        exinfo = exinfo[i, ]
      )
    }
    
    count[i + 1, ] <- count[i, ]
    count[i + 1, latent[i, ]] <- count[i + 1, latent[i, ]] + 1
  }
  
  # 删掉初始值和初始计数器
  value <- value[-1, ]
  count <- count[-1, ]

################################# [output] #####################################
    
  record@result@value       <- value
  record@result@bias        <- bias
  record@result@shown       <- shown
  record@result@prob        <- prob
  record@result@count       <- count
  
  record@result@exploration <- exploration
  record@result@latent      <- latent
  record@result@reward      <- reward
  record@result@utility     <- utility
  record@result@simulation  <- simulation

  
  output <- methods::new(
    Class = "multiRL.output",
    input = record@input,
    behrule = record@behrule,
    result = record@result,
    extra = record@extra
  )
  
  return(output)
}
