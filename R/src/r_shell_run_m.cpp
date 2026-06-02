#include <Rcpp.h>

#include <multiRL/shell_run_m.hpp>

#include "r_wrapper_common.hpp"

// [[Rcpp::export(name = ".shell_run_m")]]
Rcpp::List r_shell_run_m(
    Rcpp::CharacterMatrix object,
    Rcpp::NumericMatrix reward,
    Rcpp::CharacterVector action,
    Rcpp::IntegerVector block,
    Rcpp::IntegerVector trial,
    Rcpp::CharacterMatrix idinfo,
    Rcpp::CharacterMatrix exinfo,
    Rcpp::CharacterVector cue,
    Rcpp::CharacterVector rsp,
    Rcpp::NumericVector params,
    Rcpp::CharacterVector free_names,
    Rcpp::CharacterVector system,
    Rcpp::CharacterVector prior_names,
    Rcpp::CharacterVector prior_types,
    Rcpp::NumericVector prior_param1,
    Rcpp::NumericVector prior_param2,
    bool prior_active,
    std::string policy,
    std::string name,
    std::string mode,
    std::string estimate
) {
    multiRL::RunTask task = multiRL_r::make_task(
        object,
        reward,
        action,
        block,
        trial,
        idinfo,
        exinfo,
        cue,
        rsp,
        params,
        free_names,
        system,
        prior_names,
        prior_types,
        prior_param1,
        prior_param2,
        prior_active,
        policy,
        name,
        mode,
        estimate
    );

    multiRL::RunResult result = multiRL::shell_run_m(task);
    Rcpp::List cpp_result = multiRL_r::wrap_run_result(result, system);
    Rcpp::DataFrame fit = multiRL_r::wrap_fit(result.metric);

    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["result"] = cpp_result["result"],
        Rcpp::_["sumstat"] = fit,
        Rcpp::_["fit"] = fit
    );
    out.attr("class") = Rcpp::CharacterVector::create(
        "multiRLcpp_run_m",
        "multiRLcpp_run",
        "list"
    );
    return out;
}
