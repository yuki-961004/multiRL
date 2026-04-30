.fake_block <- function(data, block, chunk){
  
  n_block <- length(unique(data[[block]]))
  
  if (n_block == 1 | !is.null(chunk)) {
    
    n_trials <- nrow(data)
    
    # nocov start
    # 如果不能整除，寻找最近的约数进行调整
    if (n_trials %% chunk != 0) {
      divisors <- which(n_trials %% seq_len(n_trials) == 0)
      new_chunk <- divisors[which.min(abs(divisors - chunk))]
      warning(
        sprintf(
          "Since chunk does not evenly divide n_trials, it was adjusted to %d.",
          new_chunk
        ),
        call. = FALSE
      )
      chunk <- new_chunk
    }
    # nocov end
    
    # 修改block列, 将其值替换成chunk
    data[[block]] <- rep(seq_len(chunk), each = n_trials / chunk)
  }
  
  return(data)
}
