.list2df <- function(list, subid) {
  
  # 把list_simulated变成data frame
  list_data <- list()
  list_params <- list()
  
  for (i in 1:length(list)) {
    list_data[[i]] <- list[[i]]$data
    list_data[[i]][[subid]] <- i
    
    list_params[[i]] <- as.data.frame(list[[i]]$params)
    list_params[[i]][[subid]] <- i
  }
  
  df_data <- do.call(rbind, list_data)
  df_params <- do.call(rbind, list_params)
  
  dfs <- list(
    data = df_data, 
    params = df_params
  )
  
  return(dfs)
}
