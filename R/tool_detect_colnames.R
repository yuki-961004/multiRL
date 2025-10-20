.detect_colnames <- function(data, prefix) {
  
  colnames <- stringr::str_subset(
    string = names(data),
    pattern = paste0("^", prefix) 
  )
  
  if (length(colnames) == 0) {
    stop(
      "Could not automatically detect columns with prefix '",
      prefix,
      "'. Please manually specify column names."
    )
  }
  
  return(colnames)
}
