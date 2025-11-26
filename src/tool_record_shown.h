#ifndef RECORD_SHOWN_H   
#define RECORD_SHOWN_H

#include <Rcpp.h>

Rcpp::NumericVector record_shown(
    const Rcpp::CharacterMatrix& state, 
    const Rcpp::CharacterVector& cue
);

#endif