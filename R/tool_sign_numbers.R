.sign_numbers <- function(x){
  
  if(is.finite(x) && x > 0){
    sign_char <- "+"
  }
  else {
    sign_char <- ""
  }
  
  return(sign_char)
}
