#ifndef RECORD_SHOWN_HPP_   
#define RECORD_SHOWN_HPP_

#include <Rcpp.h>

Rcpp::NumericVector record_shown(
    const Rcpp::CharacterMatrix& state, 
    const Rcpp::CharacterVector& cue
);

#endif