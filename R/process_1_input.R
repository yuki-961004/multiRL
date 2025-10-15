process_1_input <- function(
  data,
  colnames,
  ...
){
  # 额外信息
  extra <- list(...)
  
  key_names <- c("subid", "block", "trial", "object", "reward", "action")
  # 检查colnames的元素名称是否一致
  check_key <- all(sort(names(colnames)) == sort(key_names))
  if (!(check_key)) {message("Invalid colnames keys")}
  # 检查colnames的元素类型是否一致
  check_type <- all(sapply(colnames, is.character))
  if (!(check_type)) {message("Invalid colnames key type")}
  
  # 如果没有输入block, 则block自动变成1
  if (is.null(colnames$block)) {data$Block = 1}
  
  methods::setClass(
    Class = "multiRL.colnames",
    slots = list(
      subid = "character", 
      block = "character",
      trial = "character",
      object = "character", 
      reward = "character",
      action = "character"
    )
  )
  
  colnames <- methods::new(
    Class = "multiRL.colnames",
    subid = colnames$subid, 
    block = colnames$block,
    trial = colnames$trial,
    object = colnames$object, 
    reward = colnames$reward,
    action = colnames$action
  )
  
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
  
  # 整合拆分后的数据, 分别是id, state和action
  methods::setClass(
    Class = "multiRL.features",
    slots = list(
      idinfo = "array",
      state = "array",
      action = "array"
    )
  )
  
  features <- methods::new(
    Class = "multiRL.features",
    idinfo = idinfo,
    state = state,
    action = action
  )
  
  # S4 method定义一个类
  methods::setClass(
    Class = "multiRL.input",
    slots = list(
      data = "data.frame",
      colnames = "multiRL.colnames",
      features = "multiRL.features",
      elements = "numeric",
      subid = "ANY",
      n_block = "numeric",
      n_trial = "numeric",
      n_rows = "numeric",
      extra = "list"
    )
  )
  
  subid <- as.character(unique(data[[colnames@subid]]))
  n_block <- length(unique(data[[colnames@block]]))
  n_trial <- length(unique(data[[colnames@trial]]))
  n_rows <- n_block * n_trial
  
  multiRL.input <- methods::new(
    Class = "multiRL.input",
    data = data,
    colnames = colnames,
    features = features,
    elements = n_element,
    subid = subid,
    n_rows = n_rows,
    n_block = n_block,
    n_trial = n_trial,
    extra = extra
  )
  
  return(multiRL.input)
}
