#include "tool_record_shown.h"

// [[Rcpp::export]]
Rcpp::IntegerVector record_shown(
    const Rcpp::CharacterMatrix& state, 
    const Rcpp::CharacterVector& cue
) {
    Rcpp::IntegerVector out(cue.size());

    for (int i = 0; i < cue.size(); ++i) {
        // 直接利用 std::find 在矩阵迭代器中查找, 避免手写循环
        // 且不创建任何额外的中间变量来存储行列数或查找状态
        out[i] = (
            std::find(state.begin(), state.end(), cue[i]) != state.end()
        ) ? 1 : 0;
    }

    return out;
}