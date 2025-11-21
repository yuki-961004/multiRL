#ifndef TOOL_SAMPLE_H
#define TOOL_SAMPLE_H

#include <Rcpp.h>

std::string sample_choice(
    const Rcpp::NumericVector& prob_row, 
    const Rcpp::NumericVector& shown_row, 
    const Rcpp::CharacterVector& cues
);

#endif