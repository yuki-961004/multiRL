#include <Rcpp.h>

#include <multiRL/estimate_rnn.hpp>

#include "r_wrapper_common.hpp"

namespace {

Rcpp::DataFrame wrap_sampler_data(
    const std::vector<multiRL::TaskSamplerResult>& results
) {
    std::size_t n_rows = 0;
    std::size_t n_prob = 0;
    std::size_t n_params = 0;
    std::vector<std::string> cue_names;
    std::vector<std::string> param_names;

    for (const auto& result : results) {
        n_rows += result.rows.size();
        if (!result.cue_names.empty()) {
            cue_names = result.cue_names;
            n_prob = cue_names.size();
        }
        if (!result.parameter_names.empty()) {
            param_names = result.parameter_names;
            n_params = param_names.size();
        }
    }

    Rcpp::IntegerVector draw(n_rows);
    Rcpp::CharacterVector subid(n_rows);
    Rcpp::IntegerVector block(n_rows);
    Rcpp::IntegerVector trial(n_rows);
    Rcpp::CharacterVector action(n_rows);
    Rcpp::CharacterVector latent(n_rows);
    Rcpp::CharacterVector simulation(n_rows);
    Rcpp::CharacterVector position(n_rows);
    Rcpp::NumericVector reward(n_rows);
    Rcpp::List prob_columns(n_prob);
    Rcpp::List param_columns(n_params);

    for (std::size_t index = 0; index < n_prob; ++index) {
        prob_columns[index] = Rcpp::NumericVector(n_rows, NA_REAL);
    }
    for (std::size_t index = 0; index < n_params; ++index) {
        param_columns[index] = Rcpp::NumericVector(n_rows, NA_REAL);
    }

    std::size_t out_row = 0;
    for (const auto& result : results) {
        for (const auto& row : result.rows) {
            draw[static_cast<R_xlen_t>(out_row)] = row.draw;
            subid[static_cast<R_xlen_t>(out_row)] = row.subid;
            block[static_cast<R_xlen_t>(out_row)] = row.block;
            trial[static_cast<R_xlen_t>(out_row)] = row.trial;
            action[static_cast<R_xlen_t>(out_row)] = row.action;
            latent[static_cast<R_xlen_t>(out_row)] = row.latent;
            simulation[static_cast<R_xlen_t>(out_row)] = row.simulation;
            position[static_cast<R_xlen_t>(out_row)] = row.position;
            reward[static_cast<R_xlen_t>(out_row)] = row.reward;

            for (std::size_t index = 0; index < n_prob; ++index) {
                Rcpp::NumericVector col = prob_columns[index];
                if (index < row.probability.size()) {
                    col[static_cast<R_xlen_t>(out_row)] =
                        row.probability[index];
                }
            }
            for (std::size_t index = 0; index < n_params; ++index) {
                Rcpp::NumericVector col = param_columns[index];
                if (index < row.params.size()) {
                    col[static_cast<R_xlen_t>(out_row)] = row.params[index];
                }
            }

            ++out_row;
        }
    }

    Rcpp::List columns = Rcpp::List::create(
        Rcpp::_["draw"] = draw,
        Rcpp::_["subid"] = subid,
        Rcpp::_["block"] = block,
        Rcpp::_["trial"] = trial,
        Rcpp::_["action"] = action,
        Rcpp::_["latent"] = latent,
        Rcpp::_["simulation"] = simulation,
        Rcpp::_["position"] = position,
        Rcpp::_["reward"] = reward
    );

    for (std::size_t index = 0; index < n_prob; ++index) {
        const std::string name = "prob_" + cue_names[index];
        columns.push_back(prob_columns[index], name);
    }
    for (std::size_t index = 0; index < n_params; ++index) {
        columns.push_back(param_columns[index], param_names[index]);
    }

    return Rcpp::DataFrame(columns);
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

    multiRL::TaskSamplerControl control;
    control.n_draws = n_draws;
    control.seed = static_cast<unsigned int>(seed);
    control.threads = threads;
    control.lower_bounds = multiRL_r::as_double_vector(lower_bounds);
    control.upper_bounds = multiRL_r::as_double_vector(upper_bounds);

    const std::vector<multiRL::TaskSamplerResult> results =
        multiRL::estimate_rnn(tasks, control);

    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["data"] = wrap_sampler_data(results),
        Rcpp::_["metadata"] = Rcpp::List::create(
            Rcpp::_["n_draws"] = control.n_draws,
            Rcpp::_["seed"] = control.seed,
            Rcpp::_["threads"] = control.threads,
            Rcpp::_["policy"] = "on",
            Rcpp::_["parameter_names"] = free_names
        )
    );
    out.attr("class") = Rcpp::CharacterVector::create(
        "multiRLcpp_estimate_rnn_data",
        "list"
    );
    return out;
}
