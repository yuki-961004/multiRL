#' @title Density and Random Function
#' @name priors
#' @description 
#'  
#'  Users must specify one of the two function types (\code{stats::?func}). 
#'    Either the Density Function (d-func) or the Random Function (r-func)
#'    
#'    These functions can represent either the sampling distribution for 
#'    generating random numbers or the prior distribution the free parameters 
#'    are assumed to follow. 
#'    
#'    Users do not need to memorize when to input the d-func or the r-func; 
#'    the program will handle the necessary conversion automatically. The 
#'    standard format for the currently supported functions will be displayed 
#'    below. Please modify only the numerical values within these functions.
#' 
#' @section Class: 
#' \code{priors [List]} 
#' 
#' @section Density Function: 
#' \preformatted{ #standard format dfunc
#'  function(x) {stats::dbeta(x, shape1 = 2, shape2 = 2, log = TRUE)}
#'  function(x) {stats::dexp(x, rate = 1, log = TRUE)}
#'  function(x) {stats::dunif(x, min = 0, max = 1, log = TRUE)}
#'  function(x) {stats::dnorm(x, mean = 0.5, sd = 0.1, log = TRUE)}
#'  function(x) {stats::dlnorm(x, meanlog = 0.5, sdlog = 0.1, log = TRUE)}
#'  function(x) {stats::dgamma(x, shape = 2, rate = 3, log = TRUE)}
#'  function(x) {stats::dlogis(x, location = 0, scale = 1, log = TRUE)}
#' }
#' 
#' @section Random Function: 
#' \preformatted{ #standard format rfunc
#'  function(x) {stats::rbeta(n = 1, shape1 = 2, shape2 = 2)}
#'  function(x) {stats::rexp(n = 1, rate = 1)}
#'  function(x) {stats::runif(n = 1, min = 0, max = 1)}
#'  function(x) {stats::rnorm(n = 1, mean = 0.5, sd = 0.1)}
#'  function(x) {stats::rlnorm(n = 1, meanlog = 0.5, sdlog = 0.1)}
#'  function(x) {stats::rgamma(n = 1, shape = 2, rate = 3)}
#'  function(x) {stats::rlogis(n = 1, location = 0, scale = 1)}
#' }
NULL