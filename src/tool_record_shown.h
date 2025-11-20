#ifndef RECORD_SHOWN_H   
#define RECORD_SHOWN_H

#include <Rcpp.h> // 必须添加 Rcpp 头文件

// 函数声明
Rcpp::IntegerVector record_shown(
    const Rcpp::CharacterMatrix& state, 
    const Rcpp::CharacterVector& cue
);

#endif