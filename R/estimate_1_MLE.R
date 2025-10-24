#' estimate_1_MLE
#'
#' @param data data
#' @param id id
#' @param behrule behrule
#' @param colnames colnames
#' @param model model
#' @param priors priors
#' @param settings settings
#' @param algorithm algorithm
#' @param lower lower
#' @param upper upper
#' @param ... extra
#'
#' @returns data.frame
#' 
estimate_1_MLE <- function(
    data, id = NULL,
    behrule,
    colnames,
    model,
    priors,
    settings,
    
    algorithm,
    lower,
    upper,
    ...
){
  
  dfinfo <- .detect_data(data)
  # 如果没有输入被试序号的列名. 则自动探测
  if ("subid" %in% names(colnames)) {subid <- colnames[["subid"]]} 
  else {subid <- dfinfo$sub_col_name}
  
  # 如果没有输入要拟合的被试序号, 就拟合所有的
  if (is.null(id)){id <- dfinfo$all_ids}
  
  # 创建空list, 用于存放结果
  env <- list()
  multiRL.model.all <- list(list(), list(), list())
  
  for (i in 1:length(model)) {
    for (j in 1:length(id)) {
      env[[i]] <- estimate_0_ENV(
        data = data[data[, subid] == j, ],
        behrule = behrule,
        colnames = colnames,
        priors = priors[[i]],
        settings = settings[[i]],
      )
      
      multiRL.model.all[[i]][[j]] <- estimate_1_LBI(
        model = model[[i]],
        env = env[[i]],
        algorithm = algorithm,
        lower = lower[[i]],
        upper = upper[[i]]
      )
    }
  }
  
  result <- .extract_results(
    multiRL.model.all, 
    n_model = length(model), n_subject = length(id)
  )
}
