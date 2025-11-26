#include "tool_record_shown.h"

// [[Rcpp::export]]
Rcpp::NumericVector record_shown(
    const Rcpp::CharacterMatrix& state, 
    const Rcpp::CharacterVector& cue
) {
    // 改用 NumericVector，此时 NA 为标准的 IEEE 754 NaN
    Rcpp::NumericVector out(cue.size());

    for (int i = 0; i < cue.size(); ++i) {
        out[i] = (
            std::find(state.begin(), state.end(), cue[i]) != state.end()
        ) ? 1.0 : NA_REAL;
    }

    return out;
}