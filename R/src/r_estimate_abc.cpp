#include <Rcpp.h>

#include <multiRL/estimate_abc.hpp>

#include "r_wrapper_common.hpp"

#include <algorithm>
#include <numeric>

namespace {

Rcpp::List wrap_estimate_abc_result(
    const std::vector<multiRL::RunTask>& tasks,
    const std::vector<multiRL::ABCSubjectResult>& results,
    const multiRL::ABCControl& control
) {
    if (results.empty()) {
        return Rcpp::List::create(
            Rcpp::_["fit"] = Rcpp::DataFrame::create(),
            Rcpp::_["estimator"] = Rcpp::List::create(
                Rcpp::_["name"] = "ABC",
                Rcpp::_["backend"] = "abcpp"
            ),
            Rcpp::_["diagnostics"] = Rcpp::List::create(
                Rcpp::_["subjects"] = Rcpp::DataFrame::create()
            )
        );
    }

    const std::vector<std::string>& free_names =
        results.front().parameter_names;
    Rcpp::CharacterVector subid(results.size());
    Rcpp::List fit_columns(free_names.size() + 2);
    Rcpp::CharacterVector fit_names(free_names.size() + 2);

    fit_columns[0] = subid;
    fit_names[0] = "subid";

    for (std::size_t col = 0; col < free_names.size(); ++col) {
        Rcpp::NumericVector values(results.size(), NA_REAL);
        for (std::size_t row = 0; row < results.size(); ++row) {
            if (col < results[row].estimates.size()) {
                values[static_cast<R_xlen_t>(row)] =
                    results[row].estimates[col];
            }
        }
        fit_columns[static_cast<R_xlen_t>(col + 1)] = values;
        fit_names[static_cast<R_xlen_t>(col + 1)] = free_names[col];
    }

    Rcpp::IntegerVector status(results.size());
    Rcpp::IntegerVector n_simulations(results.size());
    Rcpp::IntegerVector n_accepted(results.size());
    Rcpp::IntegerVector n_blocks_real(results.size());
    Rcpp::IntegerVector n_blocks_used(results.size());
    Rcpp::IntegerVector fake_block(results.size());
    Rcpp::IntegerVector n_comp_requested(results.size());
    Rcpp::IntegerVector n_comp_used(results.size());
    Rcpp::NumericVector tolerance(results.size());
    Rcpp::NumericVector min_distance(results.size(), NA_REAL);
    Rcpp::NumericVector mean_distance(results.size(), NA_REAL);
    Rcpp::CharacterVector result_message(results.size());

    Rcpp::List observed_summary(results.size());
    Rcpp::List accepted_distances(results.size());

    for (std::size_t row = 0; row < results.size(); ++row) {
        subid[static_cast<R_xlen_t>(row)] =
            row < tasks.size() ? multiRL_r::task_subid(tasks[row])
                               : results[row].subid;
        status[static_cast<R_xlen_t>(row)] = results[row].status;
        n_simulations[static_cast<R_xlen_t>(row)] =
            results[row].n_simulations;
        n_accepted[static_cast<R_xlen_t>(row)] = results[row].n_accepted;
        n_blocks_real[static_cast<R_xlen_t>(row)] =
            results[row].n_blocks_real;
        n_blocks_used[static_cast<R_xlen_t>(row)] =
            results[row].n_blocks_used;
        fake_block[static_cast<R_xlen_t>(row)] = results[row].fake_block;
        n_comp_requested[static_cast<R_xlen_t>(row)] =
            results[row].n_comp_requested;
        n_comp_used[static_cast<R_xlen_t>(row)] = results[row].n_comp_used;
        tolerance[static_cast<R_xlen_t>(row)] = control.tol;
        result_message[static_cast<R_xlen_t>(row)] = results[row].message;
        observed_summary[static_cast<R_xlen_t>(row)] =
            results[row].observed_summary;
        accepted_distances[static_cast<R_xlen_t>(row)] =
            results[row].accepted_distances;

        if (!results[row].accepted_distances.empty()) {
            const auto& distance = results[row].accepted_distances;
            min_distance[static_cast<R_xlen_t>(row)] =
                *std::min_element(distance.begin(), distance.end());
            const double total =
                std::accumulate(distance.begin(), distance.end(), 0.0);
            mean_distance[static_cast<R_xlen_t>(row)] =
                total / static_cast<double>(distance.size());
        }
    }

    const R_xlen_t status_col =
        static_cast<R_xlen_t>(free_names.size() + 1);
    fit_columns[status_col] = status;
    fit_names[status_col] = "status";
    fit_columns.names() = fit_names;

    Rcpp::List control_out = Rcpp::List::create(
        Rcpp::_["samples"] = control.samples,
        Rcpp::_["tol"] = control.tol,
        Rcpp::_["method"] = control.method,
        Rcpp::_["reduction"] = control.reduction,
        Rcpp::_["n_comp"] = control.n_comp,
        Rcpp::_["fake_block"] = control.fake_block,
        Rcpp::_["seed"] = control.seed,
        Rcpp::_["threads"] = control.threads,
        Rcpp::_["print_level"] = control.print_level
    );

    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["fit"] = Rcpp::DataFrame(fit_columns),
        Rcpp::_["estimator"] = Rcpp::List::create(
            Rcpp::_["name"] = "ABC",
            Rcpp::_["backend"] = "abcpp",
            Rcpp::_["method"] = control.method,
            Rcpp::_["reduction"] = control.reduction,
            Rcpp::_["control"] = control_out
        ),
        Rcpp::_["diagnostics"] = Rcpp::List::create(
            Rcpp::_["subjects"] = Rcpp::DataFrame::create(
                Rcpp::_["subid"] = subid,
                Rcpp::_["status"] = status,
                Rcpp::_["n_simulations"] = n_simulations,
                Rcpp::_["n_accepted"] = n_accepted,
                Rcpp::_["n_blocks_real"] = n_blocks_real,
                Rcpp::_["n_blocks_used"] = n_blocks_used,
                Rcpp::_["fake_block"] = fake_block,
                Rcpp::_["n_comp_requested"] = n_comp_requested,
                Rcpp::_["n_comp_used"] = n_comp_used,
                Rcpp::_["reduction"] = control.reduction,
                Rcpp::_["tolerance"] = tolerance,
                Rcpp::_["min_distance"] = min_distance,
                Rcpp::_["mean_distance"] = mean_distance,
                Rcpp::_["result_message"] = result_message
            ),
            Rcpp::_["abc"] = Rcpp::List::create(
                Rcpp::_["observed_summary"] = observed_summary,
                Rcpp::_["accepted_distances"] = accepted_distances
            )
        )
    );
    out.attr("class") = Rcpp::CharacterVector::create(
        "multiRLcpp_estimate_abc",
        "multiRLcpp_run",
        "list"
    );
    return out;
}

}  // namespace

// [[Rcpp::export(name = ".estimate_abc")]]
Rcpp::List r_estimate_abc(
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
    int samples,
    double tol,
    std::string method,
    std::string reduction,
    int n_comp,
    int fake_block,
    int seed,
    int threads,
    int print_level,
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
        "ABC"
    );

    multiRL::ABCControl control;
    control.samples = samples;
    control.tol = tol;
    control.method = method;
    control.reduction = reduction;
    control.reduce = reduction;
    control.n_comp = n_comp;
    control.fake_block = fake_block;
    control.seed = static_cast<unsigned int>(seed);
    control.threads = threads;
    control.print_level = print_level;
    control.lower_bounds = multiRL_r::as_double_vector(lower_bounds);
    control.upper_bounds = multiRL_r::as_double_vector(upper_bounds);

    const std::vector<multiRL::ABCSubjectResult> results =
        multiRL::estimate_abc(tasks, control);

    return wrap_estimate_abc_result(tasks, results, control);
}
