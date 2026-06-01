.modify_behrule <- function(behrule) {
  out <- utils::modifyList(list(), behrule)

  if (base::is.null(out$cue) || base::is.null(out$rsp)) {
    base::stop("behrule must contain cue and rsp.")
  }

  out
}
