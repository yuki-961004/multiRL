#include <Rcpp.h>

#ifdef MULTIRL_HAS_STAN
#include <multiRL/estimate_mcmc.hpp>
#endif

#include "r_wrapper_common.hpp"

namespace {

#ifdef MULTIRL_HAS_STAN

Rcpp::List wrap_estimate_mcmc_result(
    const std::vector<multiRL::RunTask>& tasks,
    const std::vector<multiRL::SubjectMCMCResult>& results
) {
    if (results.empty()) {
        return Rcpp::List::create(
            Rcpp::_["fit"] = Rcpp::DataFrame::create(),
            Rcpp::_["diagnostics"] = Rcpp::List::create(
                Rcpp::_["subjects"] = Rcpp::DataFrame::create()
            )
        );
    }

    const std::vector<std::string>& free_names =
        tasks[0].params.free_names;

    Rcpp::CharacterVector subid(results.size());
    Rcpp::List fit_columns(free_names.size() + 9);
    Rcpp::CharacterVector fit_names(free_names.size() + 9);

    fit_columns[0] = subid;
    fit_names[0] = "subid";

    for (std::size_t col = 0; col < free_names.size(); ++col) {
        Rcpp::NumericVector values(results.size());
        for (std::size_t row = 0; row < results.size(); ++row) {
            if (!results[row].best_params.empty() &&
                col < results[row].best_params[0].size()) {
                values[static_cast<R_xlen_t>(row)] =
                    results[row].best_params[0][col];
            } else {
                values[static_cast<R_xlen_t>(row)] = NA_REAL;
            }
        }
        fit_columns[static_cast<R_xlen_t>(col + 1)] = values;
        fit_names[static_cast<R_xlen_t>(col + 1)] = free_names[col];
    }

    Rcpp::NumericVector logL(results.size());
    Rcpp::NumericVector logPr(results.size());
    Rcpp::NumericVector logPo(results.size());
    Rcpp::NumericVector aic(results.size());
    Rcpp::NumericVector bic(results.size());
    Rcpp::IntegerVector status(results.size());
    Rcpp::IntegerVector n_evals(results.size());
    Rcpp::IntegerVector n_draws(results.size());

    for (std::size_t row = 0; row < results.size(); ++row) {
        subid[static_cast<R_xlen_t>(row)] =
            multiRL_r::task_subid(tasks[row]);
        logL[static_cast<R_xlen_t>(row)] = results[row].logL;
        logPr[static_cast<R_xlen_t>(row)] = results[row].logPrior;
        logPo[static_cast<R_xlen_t>(row)] = results[row].logPost;
        aic[static_cast<R_xlen_t>(row)] = results[row].aic;
        bic[static_cast<R_xlen_t>(row)] = results[row].bic;
        status[static_cast<R_xlen_t>(row)] = results[row].status;
        n_evals[static_cast<R_xlen_t>(row)] = results[row].n_evals;
        n_draws[static_cast<R_xlen_t>(row)] = results[row].n_draws;
    }

    const R_xlen_t offset = static_cast<R_xlen_t>(free_names.size() + 1);
    fit_columns[offset] = logL;
    fit_names[offset] = "LogL";
    fit_columns[offset + 1] = logPr;
    fit_names[offset + 1] = "LogPr";
    fit_columns[offset + 2] = logPo;
    fit_names[offset + 2] = "LogPo";
    fit_columns[offset + 3] = aic;
    fit_names[offset + 3] = "AIC";
    fit_columns[offset + 4] = bic;
    fit_names[offset + 4] = "BIC";
    fit_columns[offset + 5] = status;
    fit_names[offset + 5] = "status";
    fit_columns[offset + 6] = n_evals;
    fit_names[offset + 6] = "n_evals";
    fit_columns[offset + 7] = n_draws;
    fit_names[offset + 7] = "n_draws";
    fit_columns.names() = fit_names;

    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["fit"] = Rcpp::DataFrame(fit_columns),
        Rcpp::_["diagnostics"] = Rcpp::List::create(
            Rcpp::_["subjects"] = Rcpp::DataFrame::create(
                Rcpp::_["subid"] = subid,
                Rcpp::_["status"] = status,
                Rcpp::_["n_evals"] = n_evals,
                Rcpp::_["n_draws"] = n_draws
            )
        )
    );
    out.attr("class") = Rcpp::CharacterVector::create(
        "multiRLcpp_estimate_mcmc",
        "multiRLcpp_run",
        "list"
    );
    return out;
}

#endif // MULTIRL_HAS_STAN

}  // namespace

// [[Rcpp::export(name = ".estimate_mcmc")]]
Rcpp::List r_estimate_mcmc(
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
    int warmup,
    int samples,
    int chains,
    int thin,
    double step_size,
    double target_accept,
    int max_tree_depth,
    long seed,
    std::string algorithm,
    Rcpp::NumericVector lower_bounds,
    Rcpp::NumericVector upper_bounds
) {
#ifdef MULTIRL_HAS_STAN
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
        "MCMC"
    );

    multiRL::MCMCControl control;
    control.warmup = warmup;
    control.samples = samples;
    control.chains = chains;
    control.thin = thin;
    control.step_size = step_size;
    control.target_accept = target_accept;
    control.max_tree_depth = max_tree_depth;
    control.seed = seed;
    control.algorithm = algorithm;
    control.lower_bounds = multiRL_r::as_double_vector(lower_bounds);
    control.upper_bounds = multiRL_r::as_double_vector(upper_bounds);

    std::vector<multiRL::SubjectMCMCResult> results =
        multiRL::estimate_mcmc(tasks, control);

    return wrap_estimate_mcmc_result(tasks, results);
#else
    Rcpp::stop(
        "MCMC is not available. "
        "Stan Math headers are required for MCMC estimation."
    );
    return Rcpp::List::create();
#endif
}