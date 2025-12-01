.convert_priors <- function(priors, to = c("rfunc", "dfunc")) {
  to <- match.arg(to)
  
  convert_one <- function(fun) {
    # extract function body as text
    t <- paste(deparse(body(fun)), collapse = "")
    t <- gsub("^\\{|\\}$", "", t)  # remove braces
    
    # detect original type
    is_r <- grepl("stats::r", t)
    is_d <- grepl("stats::d", t)
    
    # extract distribution name
    dist <- sub(".*stats::[rd]([^(]+)\\(.*", "\\1", t)
    
    # extract arguments inside (...)
    args_str <- sub(".*stats::.[^(]+\\((.*)\\).*", "\\1", t)
    args <- trimws(strsplit(args_str, ",")[[1]])
    args <- args[args != ""]
    
    # -------- remove type-specific params --------
    # remove n = something
    args <- args[!grepl("^n *= *", args)]
    
    # remove log = something
    args <- args[!grepl("^log *= *", args)]
    
    # remove x OR x = something (to avoid duplicated x)
    args <- args[!grepl("^x( *=.*)?$", args)]
    
    # ---------------------------------------------
    # REBUILD as target type
    # ---------------------------------------------
    
    if (to == "rfunc") {
      new_args <- c("n = 1", args)
      fn_body <- paste0(
        "stats::r", dist, "(", paste(new_args, collapse = ", "), ")"
      )
      return(eval(parse(text = paste0("function(x) ", fn_body))))
    }
    
    if (to == "dfunc") {
      new_args <- c("x", args, "log = TRUE")
      fn_body <- paste0(
        "stats::d", dist, "(", paste(new_args, collapse = ", "), ")"
      )
      return(eval(parse(text = paste0("function(x) ", fn_body))))
    }
  }
  
  lapply(priors, function(group) {
    lapply(group, convert_one)
  })
}
