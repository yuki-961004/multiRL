#' multiRL.input
#'
#' @param data 
#'  A data frame in which each row represents a single trial,
#'    see \link[multiRL]{data} 
#' @param colnames 
#'  Column names in the data frame,
#'    see \link[multiRL]{colnames}
#' @param funcs 
#'  The functions forming the reinforcement learning model,
#'    see \link[multiRL]{funcs}
#' @param params 
#'  Parameters used by the model’s internal functions,
#'    see \link[multiRL]{params}
#' @param priors 
#'  Prior probability density function of the free parameters,
#'    see \link[multiRL]{priors}
#' @param settings 
#'  Other model settings, 
#'    see \link[multiRL]{settings}
#' @param ... 
#'  Additional arguments passed to internal functions.
#'
#' @return An S4 object of class \code{multiRL.input}.
#'
#'   \describe{
#'     \item{\code{data}}{
#'       A \code{DataFrame} containing the trial-level raw data.
#'     }
#'     \item{\code{colnames}}{
#'       An S4 object of class \code{multiRL.colnames},
#'       specifying the column names used in the input data.
#'     }
#'     \item{\code{features}}{
#'       An S4 object of class \code{multiRL.features},
#'       containing standardized representations of states and actions
#'       transformed from the raw data. 
#'     }
#'     \item{\code{params}}{
#'       An S4 object of class \code{multiRL.params}, 
#'       containing model parameters.
#'     }
#'     \item{\code{priors}}{
#'       A \code{List} specifying prior distributions for free parameters.
#'     }
#'     \item{\code{funcs}}{
#'       An S4 object of class \code{multiRL.funcs},
#'       containing functions used in model.
#'     }
#'     \item{\code{settings}}{
#'       An S4 object of class \code{multiRL.settings},
#'       storing global settings for model estimation.
#'     }
#'     \item{\code{elements}}{
#'       A \code{int} indicating the number of elements within states.
#'     }
#'     \item{\code{subid}}{
#'       A \code{Character} string identifying the subject.
#'     }
#'     \item{\code{n_block}}{
#'       A \code{int} value indicating the number of blocks.
#'     }
#'     \item{\code{n_trial}}{
#'       A \code{int} value indicating the number of trials.
#'     }
#'     \item{\code{n_rows}}{
#'       A \code{int} value indicating the number of rows in the data.
#'     }
#'     \item{\code{extra}}{
#'       A \code{List} containing additional user-defined information.
#'     }
#'   }
#'
process_1_input <- function(
    data,
    colnames = list(),
    
    funcs = list(),
    params = list(),
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
      gamma = 1, 
      delta = 0.1, 
      epsilon = NA_real_, 
      zeta = 0
    ),
    constant = list(
      seed = 123,
      L = NA_real_,
      penalty = 1,
      Q0 = NA_real_, 
      reset = NA_real_,
      lapse = 0.01,
      threshold = 1,
      bonus = 0,
      weight = 1,
      capacity = 0,
      sticky = 0
    )
  )
  # 如果一个参数在一个地方(free, fixed, constant)设置过, 则在其他地方取消
  params <- .modify_params(x = default, val = params)
  
  # 默认函数
  default <- list(
    lrng_func = multiRL::func_alpha,
    prob_func = multiRL::func_beta,
    util_func = multiRL::func_gamma,
    bias_func = multiRL::func_delta,
    expl_func = multiRL::func_epsilon,
    dcay_func = multiRL::func_zeta
  )
  funcs <- utils::modifyList(x = default, val = funcs)
  
  # 默认设置
  default <- list(
    name = "unknown",
    mode = "fitting",
    estimate = "MLE",
    policy = "on",
    system = "RL"
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
    lrng_func = funcs$lrng_func, 
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
  n_element <- nchar(
    gsub(pattern = "[^_]", replacement = "", x = object[[1]][1])
  ) + 1
  
  # func: split object based on "_"
  split_object <- function(object) {
    element <- do.call(
      what = rbind,
      args = strsplit(x = object, split = "_", fixed = TRUE)
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

  idinfo[] <- as.character(idinfo)
  object[] <- as.character(object)
  reward[] <- as.character(reward)
  action[] <- as.character(action)
  exinfo[] <- as.character(exinfo)
  
  # 整合拆分后的数据, 分别是id, state, action和其他信息
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
    policy = settings$policy,
    system = settings$system
  )
  
################################### [input] ####################################
  
  subid <- as.character(unique(data[[colnames@subid]]))
  n_block <- length(unique(data[[colnames@block]]))
  if (length(unique(as.numeric(table(data[[colnames@block]])))) == 1) {
    n_trial <- length(unique(data[[colnames@trial]]))
  } else {
    n_trial <- "varying"
  }
  n_rows <- nrow(data)
  
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
