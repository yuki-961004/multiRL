.for_abc <- function(ratio) {
  
  vector <- base::unlist(ratio)

  colname <- base::gsub(
    pattern = ".",
    replacement = "_",
    x = base::names(vector),
    fixed = TRUE
  )

  onerow <- base::matrix(
    data = vector,
    nrow = 1,
    ncol = base::length(vector),
    dimnames = base::list(NULL, colname)
  )

  return(onerow)
}
