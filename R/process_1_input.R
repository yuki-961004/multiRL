process_1_input <- function(
  data,
  colnames = list(
    subid = "Subject", 
    block = "Block",
    trial = "Trial",
    
    object = c("Object_1", "Object_2", "Object_3", "Object_4"), 
    reward = c("Reward_1", "Reward_2", "Reward_3", "Reward_4"),
    action = "Action"
  ),
  ...
){
  if (is.null(colnames$block)) {
    data$Block = 1
  }
  
  # id info
  idinfo <- as.matrix(
    data[, c(colnames$subid, colnames$block, colnames$trial)]
  )
  
  # state
  object <- as.matrix(data[, colnames$object])
  reward <- as.matrix(data[, colnames$reward])
  
  # action
  action <- as.matrix(data[, colnames$action])
  
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
  
  feature <- list(
    idinfo = idinfo,
    state = state,
    action = action
  )
  
  methods::setClass(
    Class = "multiRL.input",
    slots = list(
      data = "data.frame",
      colnames = "list",
      n_subid = "ANY",
      n_block = "ANY",
      n_trial = "ANY",
      elements = "numeric",
      features = "list"
    )
  )
  
  multiRL.input <- methods::new(
    Class = "multiRL.input",
    data = data,
    colnames = colnames,
    n_subid = unique(data[[colnames$subid]]),
    n_block = unique(data[[colnames$block]]),
    n_trial = unique(data[[colnames$trial]]),
    elements = n_element,
    features = feature
  )
  
  return(multiRL.input)
}
