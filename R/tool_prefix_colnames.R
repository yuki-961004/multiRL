.prefix_colnames <- function(x, prefix) {
  base::names(x) <- base::paste0(prefix, base::names(x))
  return(x)
}
