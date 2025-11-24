.for_abc <- function(ratio) {
  
  vector <- unlist(ratio)

  colname <- gsub(
    pattern = ".",
    replacement = "_",
    x = names(vector),
    fixed = TRUE
  )

  onerow <- matrix(
    data = vector,
    nrow = 1,
    ncol = length(vector),
    dimnames = list(NULL, colname)
  )

  return(onerow)
}
