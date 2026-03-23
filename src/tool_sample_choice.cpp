#include "../inst/include/tool_sample_choice.hpp"

// [[Rcpp::export]]
std::string sample_choice(
    const Rcpp::NumericVector& prob_row, 
    const Rcpp::NumericVector& shown_row, 
    const Rcpp::CharacterVector& cues
) {
    int n = prob_row.size();
    double total_prob = 0.0;
    int valid_count = 0;

    // 使用底层宏获取原始 C 数组指针，绕过 Rcpp::Vector 的访问开销
    const double* p_ptr = REAL(prob_row);
    const double* s_ptr = REAL(shown_row);

    // 第一遍循环: 仅计算 total_prob 并检查有效性
    // 避免了 valid_indices 和 effective_probs 的内存分配
    for (int j = 0; j < n; ++j) {
        double real_p = p_ptr[j] * s_ptr[j];
        // IEEE 754: 任何与 NaN (也就是 NA) 的比较(> 0.0)都会自动返回 false
        if (real_p > 0.0) {
            total_prob += real_p;
            valid_count++;
        }
    }

    if (valid_count == 0 || total_prob <= 0.0) {
        Rcpp::stop("No valid options to sample from.");
    }

    // 使用 R 的随机数生成器 (受 set.seed 影响)
    double rand_val = R::runif(0.0, total_prob);
    double current_sum = 0.0;
    int last_valid_idx = -1;

    // 第二遍循环: 在飞计算 (On-the-fly) 累积概率并查找
    for (int j = 0; j < n; ++j) {
        double real_p = p_ptr[j] * s_ptr[j];

        if (real_p > 0.0) {
            current_sum += real_p;
            last_valid_idx = j; // 记录以便浮点容错
            
            if (current_sum >= rand_val) {
                return Rcpp::as<std::string>(cues[j]);
            }
        }
    }

    // 浮点数容错
    return Rcpp::as<std::string>(cues[last_valid_idx]);
}