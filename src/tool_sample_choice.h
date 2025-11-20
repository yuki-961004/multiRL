#ifndef TOOL_SAMPLE_H
#define TOOL_SAMPLE_H

#include <Rcpp.h>

// [[Rcpp::plugins(cpp11)]]

/**
 * @brief 根据 (概率 * 可见性) 进行加权采样
 * * @param prob_row   原始概率向量 (Softmax 输出)
 * @param shown_row  可见性向量 (0 或 1)
 * @param cues       选项名称
 * @return std::string 被选中的 cue 名称
 */
std::string sample_choice(
    const Rcpp::NumericVector& prob_row, 
    const Rcpp::NumericVector& shown_row, 
    const Rcpp::CharacterVector& cues
);

#endif