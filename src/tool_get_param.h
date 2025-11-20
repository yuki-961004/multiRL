#ifndef GET_PARAM_H   
#define GET_PARAM_H

#include <Rcpp.h>
#include <string>

// 函数声明
double get_param(
  const Rcpp::S4& x, 
  const std::string& name
);

#endif