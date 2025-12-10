#include "../inst/include/tool_get_param.hpp" 

// [[Rcpp::export]]
double get_param(
  const Rcpp::S4& x, 
  const std::string& name
) {
  
  // 辅助 Lambda 函数：在指定槽中查找 name 并返回数值
  auto check_slot = [&](const std::string& slot_name) -> double {
    Rcpp::RObject slot_obj = x.slot(slot_name);

    if (TYPEOF(slot_obj) == VECSXP) { // 确认是 list
      Rcpp::List sublist = Rcpp::as<Rcpp::List>(slot_obj);
      if (sublist.containsElementNamed(name.c_str())) {
        Rcpp::RObject val = sublist[name];
        // 显式转换
        if (Rcpp::is<Rcpp::NumericVector>(val)) {
          Rcpp::NumericVector nv = Rcpp::as<Rcpp::NumericVector>(val);
          if (nv.size() > 0) return nv[0];
        } else if (Rcpp::is<Rcpp::IntegerVector>(val)) {
          Rcpp::IntegerVector iv = Rcpp::as<Rcpp::IntegerVector>(val);
          if (iv.size() > 0) return static_cast<double>(iv[0]);
        }
      }
    }
    return NA_REAL; // 未找到返回 NA
  };

  double found_value;

  // 1. 查找 "free"
  found_value = check_slot("free");
  if (!Rcpp::NumericVector::is_na(found_value)) return found_value;

  // 2. 查找 "fixed"
  found_value = check_slot("fixed");
  if (!Rcpp::NumericVector::is_na(found_value)) return found_value;

  // 3. 查找 "constant"
  found_value = check_slot("constant");
  if (!Rcpp::NumericVector::is_na(found_value)) return found_value;

  // 所有地方都未找到，返回 NA
  return NA_REAL;
}
