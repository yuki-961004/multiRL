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
  
  rate_func   <- record@input@funcs@rate_func
  prob_func   <- record@input@funcs@prob_func
  util_func   <- record@input@funcs@util_func
  bias_func   <- record@input@funcs@bias_func
  expl_func   <- record@input@funcs@expl_func
  dcay_func   <- record@input@funcs@dcay_func
  
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
  
  n_rows      <- record@input@n_rows
  
############################# [initial value] ##################################
  
  params      <- c(params@free, params@fixed, params@constant)
  
  seed        <- params[["seed"]]
  Q0          <- params[["Q0"]]
  reset       <- params[["reset"]]
  
  value       <- lapply(value, function(x) {
    x[1, ] <- ifelse(is.na(Q0), yes = 0, no = Q0)
    rbind(x, rep(NA, ncol(x)))
  })

  count[1, ]  <- 0
  count       <- rbind(count, rep(NA, ncol(count)))

  behave <- cbind(action, latent, simulation)
  colnames(behave) <- c("action", "latent", "simulation")
  
############################# [action select] ##################################
  
  set.seed(seed)
  
  for (i in 1:n_rows) {
    
    # 记录每个刺激是否出现
    shown[i, ] <- stats::setNames(
      object = ifelse(
        test = cue %in% state[i, , ],
        yes = 1,
        no = NA
      ),
      nm = cue
    )
    # bias function: 每个刺激上的偏见
    bias[i, ] <- bias_func(
      count = count[i, ], 
      params = params,
      idinfo = idinfo[i, ],
      exinfo = exinfo[i, ],
      behave = behave[i, ]
    )
    # exploration function: 此次是否进行探索
    exploration[i, ] <- expl_func(
      rownum = i,
      params = params,
      idinfo = idinfo[i, ],
      exinfo = exinfo[i, ],
      behave = behave[i, ]
    )

    qvalue <- lapply(value, function(x) (x[i, ] + bias[i, ]) * shown[i, ])

    # probability function: 选择每个选项的概率 
    prob[i, ] <- prob_func(
      qvalue = qvalue, 
      explor = exploration[i, ],
      params = params,
      system = system,
      idinfo = idinfo[i, ],
      exinfo = exinfo[i, ],
      behave = behave[i, ]
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
    
    behave[i, 2] <- latent[i, ]
    behave[i, 3] <- simulation[i, ]
    
############################## [value update] ##################################
    
    # 读取此时的奖励
    reward[i, ] <- state[i, row_index, dim(state)[3]]
    # utility function: 将实际奖励转化为主管价值
    utility[i, ] <- util_func(
      reward = as.numeric(reward[i, ]), 
      params = params,
      idinfo = idinfo[i, ],
      exinfo = exinfo[i, ],
      behave = behave[i, ]
    )
    
    # 判断是否需要重置：Block是否发生变化
    if (!is.na(reset)) {
      is.nb <- i > 1 && block[i] != block[i - 1]
    } else {
      is.nb <- FALSE
    }
    
    
    # 多系统更新价值
    for (sub_system in system) {
      sub_value <- value[[sub_system]]
      
      # 是否在进入新block时重置
      if (is.nb) {
        cur_value <- rep(reset, length(sub_value[i, ]))
        Qi <- reset
      } else {
        cur_value <- sub_value[i, ]
        Qi <- sub_value[i, latent[i, ]]
      }
      
      # 工作记忆容量有限导致未被选择选项的价值衰减
      sub_value[i + 1, ] <- dcay_func(
        value0 = sub_value[1, ],
        values = cur_value,
        reward = as.numeric(reward[i, ]),
        params = params,
        system = sub_system,
        idinfo = idinfo[i, ],
        exinfo = exinfo[i, ],
        behave = behave[i, ]
      )
      
      if (is.na(Q0) & Qi == 0) {
        # 如果是第一次选, 则直接记录价值 (等同于学习率100%的价值更新)
        sub_value[i + 1, latent[i, ]] <- utility[i, ]
        # 修改初始值为第一次见到的值
        sub_value[1, latent[i, ]]     <- utility[i, ]
      } else {
        # learning rate function: 如果不是第一次选, 则按照学习率方程更新
        sub_value[i + 1, latent[i, ]] <- rate_func(
          qvalue = Qi,
          reward = as.numeric(utility[i, ]),
          params = params,
          system = sub_system,
          idinfo = idinfo[i, ],
          exinfo = exinfo[i, ],
          behave = behave[i, ]
        )
      }
      
      value[[sub_system]] <- sub_value
    }
    
    count[i + 1, ] <- count[i, ]
    count[i + 1, latent[i, ]] <- count[i + 1, latent[i, ]] + 1
  }
  
  # 删掉初始值和初始计数器
  value <- lapply(value, function(x) {x[-1, ]})
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
