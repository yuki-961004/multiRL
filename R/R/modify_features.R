.modify_features <- function(data, colnames) {
  if (base::is.null(colnames$block)) {
    data$Block <- 1L
    colnames$block <- "Block"
  }

  object <- base::as.matrix(data[, colnames$object, drop = FALSE])
  reward <- base::as.matrix(data[, colnames$reward, drop = FALSE])
  action <- base::as.character(data[[colnames$action]])
  block <- base::as.integer(data[[colnames$block]])
  trial <- base::as.integer(data[[colnames$trial]])
  subid <- base::as.character(data[[colnames$subid]])

  object[] <- base::as.character(object)
  reward[] <- base::as.numeric(reward)

  idinfo <- base::cbind(
    subid = subid,
    block = base::as.character(block),
    trial = base::as.character(trial)
  )

  if (base::length(colnames$exinfo) == 1L &&
      base::is.na(colnames$exinfo)) {
    exinfo <- base::matrix(
      character(),
      nrow = base::nrow(data),
      ncol = 0L
    )
  } else {
    exinfo <- base::as.matrix(data[, colnames$exinfo, drop = FALSE])
    exinfo[] <- base::as.character(exinfo)
  }

  list(
    object = object,
    reward = reward,
    action = action,
    block = block,
    trial = trial,
    idinfo = idinfo,
    exinfo = exinfo
  )
}
