.detect_colnames <- function(data, prefix) {
  
  colnames <- stringr::str_subset(
    string = names(data),
    pattern = paste0("^", prefix) 
  )
  
  # nocov start
  if (length(colnames) == 0) {
    stop(
      "Could not automatically detect columns with prefix '",
      prefix,
      "'. Please manually specify column names."
    )
  }
  # nocov end
  
  return(colnames)
}
