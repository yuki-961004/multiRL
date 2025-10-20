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
  
################################## [check] #####################################
  
  extra <- list(...)

  if (!methods::is(record, "multiRL.record")) {
    stop("'record' must be an object of class 'multiRL.record'.")
  }
  
################################## [load] ######################################
  
  policy      <- record@input@settings@policy
  
  state       <- record@input@features@state
  action      <- record@input@features@action
  
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
    
    bias[i, ] <- bias_func(
      count = count[i, ], 
      params = params
    )
    shown[i, ] <- stats::setNames(
      object = ifelse(
        test = cue %in% state[i, , ],
        yes = 1,
        no = NA
      ),
      nm = cue
    )
    exploration[i, ] <- expl_func(
      rownum = i,
      params = params
    )
    prob[i, ] <- prob_func(
      qvalue = (value[i, ] + bias[i, ]) * shown[i, ], 
      explor = exploration[i, ],
      params = params
    )
    
    switch(
      EXPR = policy,
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
      "off" = {
        latent[i, ] <- action[i, ]
        row_index <- which(state[i, , ] == latent[i, ])
        col_index <- which(state[i, row_index, ] %in% rsp)
        simulation[i, ] <- action[i, ]
      }
    )
    
############################## [value update] ##################################
    
    reward[i, ] <- state[i, row_index, dim(state)[3]]
    utility[i, ] <- util_func(
      reward = as.numeric(reward[i, ]), 
      params = params
    )
    
    # 继承上一列的所有值
    value[i + 1, ] <- value[i, ]
    # 记录此时被选选项的值
    Qi <- value[i, latent[i, ]]
    
    if (is.na(Q1) & Qi == 0) {
      # 如果是第一次选, 则直接更新
      value[i + 1, latent[i, ]] <- utility[i, ]
    } else {
      # 如果不是第一次选, 则按照alpha方程更新
      value[i + 1, latent[i, ]] <- rate_func(
        qvalue = as.numeric(value[i, latent[i, ]]),
        reward = as.numeric(utility[i, ]), 
        params = params
      )
    }
    
    count[i + 1, ] <- count[i, ]
    count[i + 1, latent[i, ]] <- count[i + 1, latent[i, ]] + 1
  }
  
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
