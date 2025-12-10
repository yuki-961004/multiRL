#ifndef GET_PARAM_HPP_   
#define GET_PARAM_HPP_

#include <Rcpp.h>
#include <string>

double get_param(
  const Rcpp::S4& x, 
  const std::string& name
);

#endif