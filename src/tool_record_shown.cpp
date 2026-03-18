#include "../inst/include/tool_record_shown.hpp"

// [[Rcpp::export]]
Rcpp::NumericVector record_shown(
    const Rcpp::CharacterMatrix& state, 
    const Rcpp::CharacterVector& cue
) {
    // 改用 NumericVector，此时 NA 为标准的 IEEE 754 NaN
    Rcpp::NumericVector out(cue.size());

    for (int i = 0; i < cue.size(); ++i) {
        // 使用 it 接收查找结果，避免重复查找
        auto it = std::find(state.begin(), state.end(), cue[i]);

        // 若找到则通过 std::distance 计算索引并 +1，否则返回 NA
        out[i] = (it != state.end()) ? 
            (std::distance(state.begin(), it) + 1.0) : NA_REAL;
    }

    return out;
}