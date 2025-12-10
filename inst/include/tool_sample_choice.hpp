#ifndef TOOL_SAMPLE_HPP_
#define TOOL_SAMPLE_HPP_

#include <Rcpp.h>

std::string sample_choice(
    const Rcpp::NumericVector& prob_row, 
    const Rcpp::NumericVector& shown_row, 
    const Rcpp::CharacterVector& cues
);

#endif