#' multiRL.input
#'
#' @param data data
#' @param colnames colnames
#' @param params params
#' @param funcs funcs
#' @param priors priors
#' @param settings settings
#' @param ... extra
#'
#' @returns multiRL.input
#'
process_1_input <- function(
    data,
    colnames = list(),
    params = list(),
    funcs = list(),
    priors,
    settings = list(),
    ...
){
################################# [default] ####################################
  
  # 默认列名
  default <- list(
    subid = "Subject", 
    block = "Block", 
    trial = "Trial",
    object = NA_character_, 
    reward = NA_character_, 
    action = "Action",
    exinfo = NA_character_
  )
  colnames <- utils::modifyList(x = default, val = colnames)
  
  # 默认参数
  default <- list(
    free = list(),
    fixed = list(
      gamma = 1, delta = 0.1, 
      epsilon = NA_real_, zeta = 1, eta = NA_real_, theta = 0
    ),
    constant = list(
      Q1 = NA_real_, lapse = 0.01
    )
  )
  # 如果一个参数在一个地方(free, fixed, constant)设置过, 则在其他地方取消
  params <- .modify_params(x = default, val = params)
  
  # 默认函数
  default <- list(
    rate_func = multiRL::func_alpha,
    prob_func = multiRL::func_beta,
    util_func = multiRL::func_gamma,
    bias_func = multiRL::func_delta,
    expl_func = multiRL::func_epsilon,
    dcay_func = multiRL::func_theta
  )
  funcs <- utils::modifyList(x = default, val = funcs)
  
  # 默认设置
  default <- list(
    name = "unknown",
    mode = "fitting",
    estimate = "MLE",
    policy = "on"
  )
  settings <- utils::modifyList(x = default, val = settings)
  
  extra <- list(...)
  
################################# [colnames] ###################################
  
  key_names <- c(
    "subid", "block", "trial", 
    "object", "reward", "action", "exinfo"
  )
  # 检查colnames的元素名称是否一致
  check_key <- all(sort(names(colnames)) == sort(key_names))
  if (!(check_key)) {message("Invalid colnames keys")}
  
  # 检查colnames是否是字符串
  check_type <- all(sapply(colnames, is.character))
  if (!(check_type)) {message("Invalid colnames key type")}
  
################################## [idinfo] ####################################
  
  # 如果没有输入block, 则block自动变成1
  if (is.null(colnames$block)) {data$Block = 1}
  
############################# [object & reward] ################################
  
  # 如果没有设置object和reward, 可以自动探测带前缀的列
  if (length(colnames$object) == 1 && is.na(colnames$object)) {
    colnames$object <- .detect_colnames(data = data, prefix = "Object_")
  }
  if (length(colnames$reward) == 1 && is.na(colnames$reward)) {
    colnames$reward <- .detect_colnames(data = data, prefix = "Reward_")
  }
  
  colnames <- methods::new(
    Class = "multiRL.colnames",
    subid = colnames$subid, 
    block = colnames$block,
    trial = colnames$trial,
    object = colnames$object, 
    reward = colnames$reward,
    action = colnames$action,
    exinfo = colnames$exinfo
  )
  
################################## [params] ####################################
  
  # 检查params是否是数值
  check_type <- all(sapply(params$free, is.numeric))
  if (!(check_type)) {message("Invalid free params key type")}
  check_type <- all(sapply(params$fixed, is.numeric))
  if (!(check_type)) {message("Invalid fixed params key type")}
  check_type <- all(sapply(params$constant, is.numeric))
  if (!(check_type)) {message("Invalid constant params key type")}
  
  params <- methods::new(
    Class = "multiRL.params",
    free = params$free,
    fixed = params$fixed, 
    constant = params$constant
  )

################################### [funcs] ####################################

    # 检查funcs是否是函数
  check_type <- all(sapply(funcs, is.function))
  if (!(check_type)) {message("Invalid funcs key type")}
  
  funcs <- methods::new(
    Class = "multiRL.funcs",
    rate_func = funcs$rate_func, 
    prob_func = funcs$prob_func,
    util_func = funcs$util_func,
    bias_func = funcs$bias_func,
    expl_func = funcs$expl_func,
    dcay_func = funcs$dcay_func
  )
  
################################# [features] ###################################
    
  # id info
  idinfo <- as.matrix(
    data[, c(colnames@subid, colnames@block, colnames@trial)]
  )
  
  # state
  object <- as.matrix(data[, colnames@object])
  reward <- as.matrix(data[, colnames@reward])
  
  # action
  action <- as.matrix(data[, colnames@action])
  colnames(action) <- colnames@action # 单列会丢失列名
  
  # exinfo
  if (length(colnames@exinfo) == 1 && is.na(colnames@exinfo)) {
    exinfo <- base::matrix(
      data  = NA,
      nrow  = base::nrow(data),
      ncol  = 1
    )
    colnames(exinfo) <- "NullVar"
  } else {
    exinfo <- as.matrix(data[, colnames@exinfo])
    colnames(exinfo) <- colnames@exinfo
  }
  
  # object -> element
  n_element <- stringr::str_count(object[, 1][1], pattern = "_") + 1
  
  # func: split object based on "_"
  split_object <- function(object) {
    element <- stringr::str_split_fixed(
      string = object,
      pattern = "_",
      n = n_element
    )
    return(element)
  }
  
  element <- lapply(
    X = 1:ncol(object),
    FUN = function(i) {
      split_object(object[, i])
    }
  )
  
  # add reward
  state <- mapply(
    FUN = cbind, 
    element, 
    as.data.frame(reward), 
    SIMPLIFY = FALSE
  )
  
  # element: col-element-object
  state <- base::simplify2array(x = state)
  
  # element: col-object-element
  state <- base::aperm(a = state, perm = c(1, 3, 2))
  
  idinfo <- matrix(as.character(idinfo), nrow = nrow(idinfo), ncol = ncol(idinfo))
  object <- matrix(as.character(object), nrow = nrow(object), ncol = ncol(object))
  reward <- matrix(as.character(reward), nrow = nrow(reward), ncol = ncol(reward))
  action <- matrix(as.character(action), nrow = nrow(action), ncol = ncol(action))
  exinfo <- matrix(as.character(exinfo), nrow = nrow(exinfo), ncol = ncol(exinfo))

  # 整合拆分后的数据, 分别是id, state和action
  features <- methods::new(
    Class = "multiRL.features",
    idinfo = idinfo,
    state = state,
    action = action,
    exinfo = exinfo
  )

################################# [settings] ###################################
  
  check_type <- all(sapply(settings, is.character))
  if (!(check_type)) {message("Invalid settings key type")}
  
  settings <- methods::new(
    Class = "multiRL.settings",
    name = settings$name,
    mode = settings$mode,
    estimate = settings$estimate,
    policy = settings$policy
  )
  
################################### [input] ####################################
  
  subid <- as.character(unique(data[[colnames@subid]]))
  n_block <- length(unique(data[[colnames@block]]))
  n_trial <- length(unique(data[[colnames@trial]]))
  n_rows <- n_block * n_trial
  
  multiRL.input <- methods::new(
    Class = "multiRL.input",
    data = data,
    colnames = colnames,
    features = features,
    params = params,
    funcs = funcs,
    priors = priors,
    settings = settings,
    elements = n_element,
    subid = subid,
    n_rows = n_rows,
    n_block = n_block,
    n_trial = n_trial,
    extra = extra
  )
  
  return(multiRL.input)
}
