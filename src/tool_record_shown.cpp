#include "tool_record_shown.h"

// [[Rcpp::export]]
Rcpp::IntegerVector record_shown(
    const Rcpp::CharacterMatrix& state, 
    const Rcpp::CharacterVector& cue
) {
    Rcpp::IntegerVector out(cue.size());

    for (int i = 0; i < cue.size(); ++i) {
        out[i] = (
            std::find(state.begin(), state.end(), cue[i]) != state.end()
        ) ? 1 : 0;
    }

    return out;
}