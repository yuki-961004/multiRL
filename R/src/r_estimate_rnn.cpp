#include <Rcpp.h>

#include <multiRL/estimate_rnn.hpp>

#include "r_wrapper_common.hpp"

namespace {

Rcpp::DataFrame wrap_rnn_fit(
    const std::vector<multiRL::EstimateRnnSubjectResult>& results
) {
    std::size_t n_rows = results.size();
    std::vector<std::string> param_names;
    for (const auto& result : results) {
        if (!result.parameter_names.empty()) {
            param_names = result.parameter_names;
            break;
        }
    }

    Rcpp::CharacterVector subid(n_rows);
    Rcpp::IntegerVector status(n_rows);
    Rcpp::IntegerVector n_draws(n_rows);
    Rcpp::IntegerVector n_trials(n_rows);
    Rcpp::IntegerVector n_features(n_rows);
    Rcpp::NumericVector loss(n_rows);
    Rcpp::CharacterVector message(n_rows);
    Rcpp::List param_columns(param_names.size());

    for (std::size_t index = 0; index < param_names.size(); ++index) {
        param_columns[index] = Rcpp::NumericVector(n_rows, NA_REAL);
    }

    for (std::size_t row = 0; row < n_rows; ++row) {
        const auto& result = results[row];
        subid[static_cast<R_xlen_t>(row)] = result.subid;
        status[static_cast<R_xlen_t>(row)] = result.status;
        n_draws[static_cast<R_xlen_t>(row)] = result.n_draws;
        n_trials[static_cast<R_xlen_t>(row)] = result.n_trials;
        n_features[static_cast<R_xlen_t>(row)] = result.n_features;
        loss[static_cast<R_xlen_t>(row)] = result.loss;
        message[static_cast<R_xlen_t>(row)] = result.message;

        for (std::size_t index = 0; index < param_names.size(); ++index) {
            Rcpp::NumericVector col = param_columns[index];
            if (index < result.estimates.size()) {
                col[static_cast<R_xlen_t>(row)] = result.estimates[index];
            }
        }
    }

    Rcpp::List columns = Rcpp::List::create(
        Rcpp::_["subid"] = subid
    );

    for (std::size_t index = 0; index < param_names.size(); ++index) {
        columns.push_back(param_columns[index], param_names[index]);
    }

    columns.push_back(status, "status");
    columns.push_back(n_draws, "n_draws");
    columns.push_back(n_trials, "n_trials");
    columns.push_back(n_features, "n_features");
    columns.push_back(loss, "loss");
    columns.push_back(message, "message");

    return Rcpp::DataFrame(columns);
}

Rcpp::DataFrame wrap_rnn_diagnostics(
    const std::vector<multiRL::EstimateRnnSubjectResult>& results
) {
    std::size_t n_rows = results.size();
    Rcpp::CharacterVector subid(n_rows);
    Rcpp::IntegerVector status(n_rows);
    Rcpp::IntegerVector n_draws(n_rows);
    Rcpp::IntegerVector n_trials(n_rows);
    Rcpp::IntegerVector n_features(n_rows);
    Rcpp::IntegerVector epochs(n_rows);
    Rcpp::NumericVector loss(n_rows);
    Rcpp::CharacterVector architecture(n_rows);
    Rcpp::CharacterVector message(n_rows);

    for (std::size_t row = 0; row < n_rows; ++row) {
        const auto& result = results[row];
        subid[static_cast<R_xlen_t>(row)] = result.subid;
        status[static_cast<R_xlen_t>(row)] = result.status;
        n_draws[static_cast<R_xlen_t>(row)] = result.n_draws;
        n_trials[static_cast<R_xlen_t>(row)] = result.n_trials;
        n_features[static_cast<R_xlen_t>(row)] = result.n_features;
        epochs[static_cast<R_xlen_t>(row)] = result.epochs;
        loss[static_cast<R_xlen_t>(row)] = result.loss;
        architecture[static_cast<R_xlen_t>(row)] = result.architecture;
        message[static_cast<R_xlen_t>(row)] = result.message;
    }

    return Rcpp::DataFrame::create(
        Rcpp::_["subid"] = subid,
        Rcpp::_["status"] = status,
        Rcpp::_["n_draws"] = n_draws,
        Rcpp::_["n_trials"] = n_trials,
        Rcpp::_["n_features"] = n_features,
        Rcpp::_["epochs"] = epochs,
        Rcpp::_["loss"] = loss,
        Rcpp::_["architecture"] = architecture,
        Rcpp::_["message"] = message
    );
}

}  // namespace

// [[Rcpp::export(name = ".estimate_rnn_data")]]
Rcpp::List r_estimate_rnn_data(
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
    int n_draws,
    int seed,
    int threads,
    int epochs,
    int batch_size,
    int units,
    int layers,
    double dropout,
    double learning_rate,
    std::string model_type,
    int verbose,
    std::string device,
    std::string scope,
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
        "RNN"
    );

    multiRL::RNNControl control;
    control.n_draws = n_draws;
    control.sample = n_draws;
    control.seed = seed;
    control.threads = threads;
    control.epoch = epochs;
    control.epochs = epochs;
    control.batch_size = batch_size;
    control.units = units;
    control.layers = layers;
    control.dropout = dropout;
    control.learning_rate = learning_rate;
    control.model_type = model_type;
    control.verbose = verbose;
    control.device = device;
    control.scope = scope;
    control.lower_bounds = multiRL_r::as_double_vector(lower_bounds);
    control.upper_bounds = multiRL_r::as_double_vector(upper_bounds);

    const std::vector<multiRL::EstimateRnnSubjectResult> results =
        multiRL::estimate_rnn(tasks, control);

    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["fit"] = wrap_rnn_fit(results),
        Rcpp::_["estimator"] = Rcpp::List::create(
            Rcpp::_["name"] = "RNN",
            Rcpp::_["backend"] = "torch",
            Rcpp::_["architecture"] = model_type
        ),
        Rcpp::_["diagnostics"] = Rcpp::List::create(
            Rcpp::_["subjects"] = wrap_rnn_diagnostics(results)
        ),
        Rcpp::_["metadata"] = Rcpp::List::create(
            Rcpp::_["n_draws"] = control.n_draws,
            Rcpp::_["seed"] = control.seed,
            Rcpp::_["threads"] = control.threads,
            Rcpp::_["backend"] = "torch",
            Rcpp::_["parameter_names"] = free_names
        )
    );
    out.attr("class") = Rcpp::CharacterVector::create(
        "multiRLcpp_estimate_rnn_data",
        "list"
    );
    return out;
}
