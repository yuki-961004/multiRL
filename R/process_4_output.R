#' multiRL.output
#'
#' @param record 
#'  multiRL.record
#' @param ... 
#'  Additional arguments passed to internal functions.
#'  
#' @return An S4 object of class \code{multiRL.output}.
#' 
#'   \describe{
#'     \item{\code{input}}{
#'       An object of class \code{multiRL.input},
#'       containing the raw data, column specifications, parameters and ...
#'     }
#'     \item{\code{behrule}}{
#'       An object of class \code{multiRL.behrule},
#'       defining the latent learning rules.
#'     }
#'     \item{\code{result}}{
#'       An object of class \code{multiRL.result},
#'       storing trial-level outputs of the Markov Decision Process.
#'     }
#'     \item{\code{extra}}{
#'       A \code{List} containing additional user-defined information.
#'     }
#'   }
#' 
process_4_output_r <- function(
    record,
    ...
){
  
  extra <- list(...)
  
################################## [load] ######################################
  
  policy      <- record@input@settings@policy
  system      <- record@input@settings@system
  
  idinfo      <- record@input@features@idinfo
  subid       <- idinfo[, 1]
  block       <- idinfo[, 2]
  trial       <- idinfo[, 3]
  
  state       <- record@input@features@state
  action      <- record@input@features@action
  exinfo      <- record@input@features@exinfo
  
  params      <- record@input@params
  
  lrng_func   <- record@input@funcs@lrng_func
  prob_func   <- record@input@funcs@prob_func
  util_func   <- record@input@funcs@util_func
  bias_func   <- record@input@funcs@bias_func
  expl_func   <- record@input@funcs@expl_func
  dcay_func   <- record@input@funcs@dcay_func
  
  cue         <- record@behrule@cue
  mid         <- record@behrule@mid
  rsp         <- record@behrule@rsp
  
  value       <- record@result@value
  bias        <- record@result@bias
  utility     <- record@result@utility
  shown       <- record@result@shown
  prob        <- record@result@prob
  count       <- record@result@count
  
  hidden      <- record@result@hidden

  exploration <- record@result@exploration
  latent      <- record@result@latent
  reward      <- record@result@reward
  simulation  <- record@result@simulation
  position    <- record@result@position
  
  n_rows      <- record@input@n_rows
  
############################# [initial value] ##################################
  
  params      <- c(params@free, params@fixed, params@constant)
  
  seed        <- params[["seed"]]
  Q0          <- params[["Q0"]]
  reset       <- params[["reset"]]
  
  value       <- lapply(value, function(x) {
    x[1, ] <- ifelse(is.nan(Q0), yes = 0, no = Q0)
    rbind(x, rep(NA_real_, ncol(x)))
  })

  count[1, ]  <- 0
  count       <- rbind(count, rep(NA_real_, ncol(count)))

  behave <- cbind(action, latent, simulation, position)
  behave <- rbind(behave, rep(NA_character_, ncol(behave)))
  colnames(behave) <- c("action", "latent", "simulation", "position")
  
  hidden <- rbind(hidden, rep(NA_character_, ncol(hidden)))
  colnames(hidden) <- mid

############################# [action select] ##################################
  
  set.seed(seed)
  
  for (i in 1:n_rows) {
    
    # 记录每个刺激是否出现
    shown[i, ] <- stats::setNames(
      object = base::match(
        x = cue, 
        table = state[i, , ]
      ),
      nm = cue
    )
    # bias function: 每个刺激上的偏见
    bias_results <- bias_func( 
      shown = shown[i, ],
      count = count[i, ], 

      rownum = i,
      params = params,
      hidden = hidden[i, ], 

      idinfo = idinfo[i, ],
      exinfo = exinfo[i, ],
      behave = behave[i, ],
      cue = cue, rsp = rsp,
      state = state[i, , ]
    )
    bias[i, ] <- bias_results$output 
    hidden[i, ] <- bias_results$hidden 
    hidden[i + 1, ] <- bias_results$hidden 

    # exploration function: 此次是否进行探索
    expl_results <- expl_func(
      shown = shown[i, ],

      rownum = i,
      params = params,
      hidden = hidden[i, ],

      idinfo = idinfo[i, ],
      exinfo = exinfo[i, ],
      behave = behave[i, ],
      cue = cue, rsp = rsp,
      state = state[i, , ]
    )
    exploration[i, ] <- expl_results$output
    hidden[i, ] <- expl_results$hidden 
    hidden[i + 1, ] <- expl_results$hidden 

    qvalue <- lapply(value, function(x) {
      v <- x[i, ] + bias[i, ]
      # 如果该试次不出现某选项, 则替换成NA_real_
      v[is.na(shown[i, ])] <- NA_real_
      # 如果该试次出现的选项价值都是NA, 则说明Q0 = NA_real_, 第零行Q需要替换为0
      v[!is.na(shown[i, ]) & is.na(v)] <- 0
      return(v)
    })

    # probability function: 选择每个选项的概率 
    prob_results <- prob_func(
      shown = shown[i, ],
      qvalue = qvalue, 
      explor = exploration[i, ],
      system = system,

      rownum = i,
      params = params,
      hidden = hidden[i, ],
      
      idinfo = idinfo[i, ],
      exinfo = exinfo[i, ],
      behave = behave[i, ],
      cue = cue, rsp = rsp,
      state = state[i, , ]
    )
    prob[i, ] <- prob_results$output
    hidden[i, ] <- prob_results$hidden 
    hidden[i + 1, ] <- prob_results$hidden 

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

    position[i, ] <- as.character(row_index)
    # 记录当前行为到当前试次, 会覆盖上一次的行为
    behave[i, 2] <- latent[i, ]
    behave[i, 3] <- simulation[i, ]
    behave[i, 4] <- position[i, ]
    # 记录当前行为到下一个试次, 用于action select三函数读取
    behave[i + 1, 2] <- latent[i, ]
    behave[i + 1, 3] <- simulation[i, ]
    behave[i + 1, 4] <- position[i, ]
    
############################## [value update] ##################################
    
    # 读取此时的奖励
    reward[i, ] <- state[i, row_index, dim(state)[3]]
    # utility function: 将实际奖励转化为主管价值
    util_results <- util_func(
      shown = shown[i, ],
      reward = as.numeric(reward[i, ]), 

      rownum = i,
      params = params,
      hidden = hidden[i, ],

      idinfo = idinfo[i, ],
      exinfo = exinfo[i, ],
      behave = behave[i, ],
      cue = cue, rsp = rsp,
      state = state[i, , ]
    )
    utility[i, ] <- util_results$output 
    hidden[i, ] <- util_results$hidden 
    hidden[i + 1, ] <- util_results$hidden 
    
    # 判断是否需要重置：Block是否发生变化
    is.nb <- trial[i] == 1
    # 检查此时是否是第一次选(全局第一次 or 局部第一次, 都算)
    is.fp <- count[i, latent[i, ]] == 0

    # 多系统更新价值
    for (sub_system in system) {
      
      sub_value <- value[[sub_system]]
      
      # 工作记忆容量有限导致未被选择选项的价值衰减
      dcay_results <- dcay_func(
        shown = shown[i, ],
        is.nb = is.nb,
        value0 = sub_value[1, ],
        values = sub_value[i, ],
        reward = as.numeric(reward[i, ]),
        utility = as.numeric(utility[i, ]),
        system = sub_system,

        rownum = i,
        params = params,
        hidden = hidden[i, ],
        
        idinfo = idinfo[i, ],
        exinfo = exinfo[i, ],
        behave = behave[i, ],
        cue = cue, rsp = rsp,
        state = state[i, , ]
      )
      sub_value[i + 1, ] <- dcay_results$output 
      hidden[i, ] <- dcay_results$hidden 
      hidden[i + 1, ] <- dcay_results$hidden 

      # 从当前行读取Qi
      Qi = sub_value[i, latent[i, ]]
      # 如果是新block, 且reset是NA_real而非NaN, 则Qi从新block第一试次读取. 
      if (is.nb && is.na(reset) && !is.nan(reset)) {
        Qi = sub_value[i + 1, latent[i, ]]
      }

      # learning rate function: 如果不是第一次选, 则按照学习率方程更新
      lrng_results <- lrng_func(
        shown = shown[i, ],
        is.fp = is.fp,
        qvalue = Qi,
        reward = as.numeric(reward[i, ]),
        utility = as.numeric(utility[i, ]),
        system = sub_system,

        rownum = i,
        params = params,
        hidden = hidden[i, ],
        
        idinfo = idinfo[i, ],
        exinfo = exinfo[i, ],
        behave = behave[i, ],
        cue = cue, rsp = rsp,
        state = state[i, , ]
      )
      sub_value[i + 1, latent[i, ]] <- lrng_results$output 
      hidden[i, ] <- lrng_results$hidden 
      hidden[i + 1, ] <- lrng_results$hidden 

      # 如果Q0为NaN, 且是第一次选, 进行了100%学习率价值更新
      if (is.nan(Q0) && is.fp) {
        # 将初始值替换为第一次见到的值
        sub_value[1, latent[i, ]] <- lrng_results$output
      }
      
      value[[sub_system]] <- sub_value
    }
    
    # 如果需要重置, 且进入了新block, 则计数器也要归零
    if (is.nb && is.nan(reset)) {
      count[i + 1, ] <- 0
    } else {
      count[i + 1, ] <- count[i, ]
    }
    
    count[i + 1, latent[i, ]] <- count[i + 1, latent[i, ]] + 1
  }
  
################################# [output] #####################################
    
  record@result@value       <- value
  record@result@bias        <- bias
  record@result@shown       <- shown
  record@result@prob        <- prob
  record@result@count       <- count
  
  record@result@hidden      <- hidden

  record@result@exploration <- exploration
  record@result@latent      <- latent
  record@result@reward      <- reward
  record@result@utility     <- utility
  record@result@simulation  <- simulation
  record@result@position    <- position

  output <- methods::new(
    Class = "multiRL.output",
    input = record@input,
    behrule = record@behrule,
    result = record@result,
    extra = record@extra
  )
  
  return(output)
}
