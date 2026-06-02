#include <Rcpp.h>

#include <multiRL/estimate_map.hpp>

#include "r_wrapper_common.hpp"

namespace {

Rcpp::List wrap_estimate_map_result(
    const std::vector<multiRL::RunTask>& tasks,
    const multiRL::EstimateMapResult& result
) {
    const std::vector<multiRL::EstimateMleResult>& results = result.subjects;

    if (results.empty()) {
        return Rcpp::List::create(
            Rcpp::_["fit"] = Rcpp::DataFrame::create(),
            Rcpp::_["diagnostics"] = Rcpp::List::create(
                Rcpp::_["subjects"] = Rcpp::DataFrame::create(),
                Rcpp::_["em"] = Rcpp::DataFrame::create()
            )
        );
    }

    const std::vector<std::string>& free_names = results[0].params.free_names;
    Rcpp::CharacterVector subid(results.size());
    Rcpp::List fit_columns(free_names.size() + 10);
    Rcpp::CharacterVector fit_names(free_names.size() + 10);

    fit_columns[0] = subid;
    fit_names[0] = "subid";

    for (std::size_t col = 0; col < free_names.size(); ++col) {
        Rcpp::NumericVector values(results.size());
        for (std::size_t row = 0; row < results.size(); ++row) {
            values[static_cast<R_xlen_t>(row)] =
                results[row].params.get(free_names[col]);
        }
        fit_columns[static_cast<R_xlen_t>(col + 1)] = values;
        fit_names[static_cast<R_xlen_t>(col + 1)] = free_names[col];
    }

    Rcpp::NumericVector acc(results.size());
    Rcpp::NumericVector log_likelihood(results.size());
    Rcpp::NumericVector log_prior(results.size());
    Rcpp::NumericVector log_posterior(results.size());
    Rcpp::NumericVector aic(results.size());
    Rcpp::NumericVector bic(results.size());
    Rcpp::IntegerVector status(results.size());
    Rcpp::IntegerVector n_evals(results.size());
    Rcpp::NumericVector optimum_value(results.size());
    Rcpp::CharacterVector result_message(results.size());
    Rcpp::CharacterVector stop_reason(results.size());

    for (std::size_t row = 0; row < results.size(); ++row) {
        subid[static_cast<R_xlen_t>(row)] = multiRL_r::task_subid(tasks[row]);
        acc[static_cast<R_xlen_t>(row)] = results[row].metric.acc;
        log_likelihood[static_cast<R_xlen_t>(row)] =
            results[row].metric.log_likelihood;
        log_prior[static_cast<R_xlen_t>(row)] = results[row].metric.log_prior;
        log_posterior[static_cast<R_xlen_t>(row)] =
            results[row].metric.log_posterior;
        aic[static_cast<R_xlen_t>(row)] = results[row].metric.aic;
        bic[static_cast<R_xlen_t>(row)] = results[row].metric.bic;
        status[static_cast<R_xlen_t>(row)] = results[row].status;
        n_evals[static_cast<R_xlen_t>(row)] = results[row].n_evals;
        optimum_value[static_cast<R_xlen_t>(row)] =
            results[row].optimum_value;
        result_message[static_cast<R_xlen_t>(row)] =
            results[row].result_message;
        stop_reason[static_cast<R_xlen_t>(row)] = results[row].stop_reason;
    }

    const R_xlen_t offset = static_cast<R_xlen_t>(free_names.size() + 1);
    fit_columns[offset] = acc;
    fit_names[offset] = "ACC";
    fit_columns[offset + 1] = log_likelihood;
    fit_names[offset + 1] = "LogL";
    fit_columns[offset + 2] = log_prior;
    fit_names[offset + 2] = "LogPr";
    fit_columns[offset + 3] = log_posterior;
    fit_names[offset + 3] = "LogPo";
    fit_columns[offset + 4] = aic;
    fit_names[offset + 4] = "AIC";
    fit_columns[offset + 5] = bic;
    fit_names[offset + 5] = "BIC";
    fit_columns[offset + 6] = status;
    fit_names[offset + 6] = "status";
    fit_columns[offset + 7] = n_evals;
    fit_names[offset + 7] = "n_evals";
    fit_columns[offset + 8] = optimum_value;
    fit_names[offset + 8] = "optimum_value";
    fit_columns.names() = fit_names;

    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["fit"] = Rcpp::DataFrame(fit_columns),
        Rcpp::_["diagnostics"] = Rcpp::List::create(
            Rcpp::_["subjects"] = Rcpp::DataFrame::create(
                Rcpp::_["subid"] = subid,
                Rcpp::_["status"] = status,
                Rcpp::_["n_evals"] = n_evals,
                Rcpp::_["optimum_value"] = optimum_value,
                Rcpp::_["result_message"] = result_message,
                Rcpp::_["stop_reason"] = stop_reason
            ),
            Rcpp::_["em"] = Rcpp::DataFrame::create(
                Rcpp::_["iterations"] = result.iterations,
                Rcpp::_["delta"] = result.delta,
                Rcpp::_["best_log_posterior"] =
                    result.best_log_posterior,
                Rcpp::_["stop_reason"] = result.stop_reason
            )
        )
    );
    out.attr("class") = Rcpp::CharacterVector::create(
        "multiRLcpp_estimate_map",
        "multiRLcpp_run",
        "list"
    );
    return out;
}

}  // namespace

// [[Rcpp::export(name = ".estimate_map")]]
Rcpp::List r_estimate_map(
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
    int mle_maxeval,
    int map_maxiter,
    double map_tol,
    int map_patience,
    std::string algorithm,
    std::string local_algorithm,
    double xtol_rel,
    double local_xtol_rel,
    int seed,
    Rcpp::NumericVector lower_bounds,
    Rcpp::NumericVector upper_bounds
) {
    std::vector<multiRL::RunTask> tasks = multiRL_r::make_subject_tasks(
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
        "MAP"
    );

    multiRL::MAPControl control;
    control.mle.nlopt.maxeval = mle_maxeval;
    control.mle.nlopt.algorithm = algorithm;
    control.mle.nlopt.local_algorithm = local_algorithm;
    control.mle.nlopt.xtol_rel = xtol_rel;
    control.mle.nlopt.local_xtol_rel = local_xtol_rel;
    control.mle.nlopt.seed = seed;
    control.mle.nlopt.lower_bounds =
        multiRL_r::as_double_vector(lower_bounds);
    control.mle.nlopt.upper_bounds =
        multiRL_r::as_double_vector(upper_bounds);
    control.map_maxiter = map_maxiter;
    control.map_tol = map_tol;
    control.map_patience = map_patience;

    multiRL::EstimateMapResult result = multiRL::estimate_map(tasks, control);
    return wrap_estimate_map_result(tasks, result);
}
