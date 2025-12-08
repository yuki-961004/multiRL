.make_empty_value <- function(class_name) {
  switch(
    class_name,
    "character"   = base::character(),
    "numeric"     = base::numeric(),
    "integer"     = base::integer(),
    "logical"     = base::logical(),
    "list"        = base::list(),
    "data.frame"  = base::data.frame(),
    "array"       = base::array(),
    "function"    = base::`function`(),
    # 对于自定义 S4 类：创建一个空对象
    methods::new(class_name)
  )
}

.remove_slot <- function(x, omit = base::character()) {
  
  #------ S4 对象 ------
  if (base::isS4(x)) {
    slot_classes <- methods::getSlots(class(x))
    
    for (slot_name in base::names(slot_classes)) {
      if (slot_name %in% omit) {
        empty_val <- .make_empty_value(slot_classes[[slot_name]])
        methods::slot(x, slot_name) <- empty_val
      } else {
        inner <- methods::slot(x, slot_name)
        methods::slot(x, slot_name) <- .remove_slot(inner, omit = omit)
      }
    }
    return(x)
  }
  
  #------ 普通 list 与 S3 ------
  if (!base::is.list(x)) return(x)
  
  slot_name <- base::names(x)
  if (!base::is.null(slot_name)) {
    drop <- slot_name %in% omit
    x[slot_name[drop]] <- NULL
  }
  
  for (i in base::seq_along(x)) {
    x[[i]] <- .remove_slot(x[[i]], omit = omit)
  }
  
  return(x)
}
