.modify_colnames <- function(data, colnames) {
  default_colnames <- list(
    subid = "Subject",
    block = "Block",
    trial = "Trial",
    object = NA_character_,
    reward = NA_character_,
    action = "Action",
    exinfo = NA_character_
  )
  out <- utils::modifyList(default_colnames, colnames)

  if (base::length(out$object) == 1L && base::is.na(out$object)) {
    out$object <- base::grep(
      "^Object_",
      base::names(data),
      value = TRUE
    )
  }

  if (base::length(out$reward) == 1L && base::is.na(out$reward)) {
    out$reward <- base::grep(
      "^Reward_",
      base::names(data),
      value = TRUE
    )
  }

  out
}

.modify_data_id <- function(data, id, subid) {
  if (!base::is.null(id)) {
    data <- data[data[[subid]] %in% id, , drop = FALSE]
  }

  if (base::nrow(data) == 0L) {
    base::stop("No data rows remain after id filtering.")
  }

  data
}
