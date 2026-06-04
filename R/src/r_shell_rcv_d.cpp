#include <Rcpp.h>

#include <algorithm>

#include <multiRL/shell_rcv_d.hpp>

#include "r_wrapper_common.hpp"

namespace {

Rcpp::DataFrame wrap_recovery_simulation(
    const multiRL::RecoveryResult& result
) {
    std::size_t n_object = 0;
    std::size_t n_reward = 0;
    std::size_t n_prob = result.cue_names.size();

    for (const auto& row : result.simulation) {
        n_object = std::max(n_object, row.object.size());
        n_reward = std::max(n_reward, row.reward_table.size());
    }

    const std::size_t n_rows = result.simulation.size();
    Rcpp::CharacterVector generating_model(n_rows);
    Rcpp::IntegerVector draw(n_rows);
    Rcpp::CharacterVector subid(n_rows);
    Rcpp::IntegerVector block(n_rows);
    Rcpp::IntegerVector trial(n_rows);
    Rcpp::CharacterVector action(n_rows);
    Rcpp::CharacterVector latent(n_rows);
    Rcpp::CharacterVector simulation(n_rows);
    Rcpp::CharacterVector position(n_rows);
    Rcpp::NumericVector reward(n_rows);
    Rcpp::List object_columns(n_object);
    Rcpp::List reward_columns(n_reward);
    Rcpp::List prob_columns(n_prob);

    for (std::size_t index = 0; index < n_object; ++index) {
        object_columns[index] = Rcpp::CharacterVector(n_rows);
    }
    for (std::size_t index = 0; index < n_reward; ++index) {
        reward_columns[index] = Rcpp::NumericVector(n_rows, NA_REAL);
    }
    for (std::size_t index = 0; index < n_prob; ++index) {
        prob_columns[index] = Rcpp::NumericVector(n_rows, NA_REAL);
    }

    for (std::size_t row_index = 0; row_index < n_rows; ++row_index) {
        const auto& row = result.simulation[row_index];
        const R_xlen_t out_row = static_cast<R_xlen_t>(row_index);
        generating_model[out_row] = row.generating_model;
        draw[out_row] = row.draw;
        subid[out_row] = row.subid;
        block[out_row] = row.block;
        trial[out_row] = row.trial;
        action[out_row] = row.action;
        latent[out_row] = row.latent;
        simulation[out_row] = row.simulation;
        position[out_row] = row.position;
        reward[out_row] = row.reward;

        for (std::size_t index = 0; index < n_object; ++index) {
            Rcpp::CharacterVector col = object_columns[index];
            if (index < row.object.size()) {
                col[out_row] = row.object[index];
            }
        }
        for (std::size_t index = 0; index < n_reward; ++index) {
            Rcpp::NumericVector col = reward_columns[index];
            if (index < row.reward_table.size()) {
                col[out_row] = row.reward_table[index];
            }
        }
        for (std::size_t index = 0; index < n_prob; ++index) {
            Rcpp::NumericVector col = prob_columns[index];
            if (index < row.probability.size()) {
                col[out_row] = row.probability[index];
            }
        }
    }

    Rcpp::List columns = Rcpp::List::create(
        Rcpp::_["generating_model"] = generating_model,
        Rcpp::_["draw"] = draw,
        Rcpp::_["subid"] = subid,
        Rcpp::_["block"] = block,
        Rcpp::_["trial"] = trial
    );

    for (std::size_t index = 0; index < n_object; ++index) {
        columns.push_back(
            object_columns[index],
            "object_" + std::to_string(index + 1)
        );
    }
    for (std::size_t index = 0; index < n_reward; ++index) {
        columns.push_back(
            reward_columns[index],
            "reward_" + std::to_string(index + 1)
        );
    }

    columns.push_back(action, "action");
    columns.push_back(latent, "latent");
    columns.push_back(simulation, "simulation");
    columns.push_back(position, "position");
    columns.push_back(reward, "sampled_reward");

    for (std::size_t index = 0; index < n_prob; ++index) {
        columns.push_back(
            prob_columns[index],
            "prob_" + result.cue_names[index]
        );
    }

    return Rcpp::DataFrame(columns);
}

Rcpp::DataFrame wrap_recovery_truth(
    const multiRL::RecoveryResult& result
) {
    const std::size_t n_rows = result.truth.size();
    const std::size_t n_params = result.parameter_names.size();
    Rcpp::CharacterVector generating_model(n_rows);
    Rcpp::IntegerVector draw(n_rows);
    Rcpp::CharacterVector subid(n_rows);
    Rcpp::List columns = Rcpp::List::create(
        Rcpp::_["generating_model"] = generating_model,
        Rcpp::_["draw"] = draw,
        Rcpp::_["subid"] = subid
    );
    Rcpp::List param_columns(n_params);

    for (std::size_t index = 0; index < n_params; ++index) {
        param_columns[index] = Rcpp::NumericVector(n_rows, NA_REAL);
    }

    for (std::size_t row_index = 0; row_index < n_rows; ++row_index) {
        const auto& row = result.truth[row_index];
        const R_xlen_t out_row = static_cast<R_xlen_t>(row_index);
        generating_model[out_row] = row.generating_model;
        draw[out_row] = row.draw;
        subid[out_row] = row.subid;

        for (std::size_t index = 0; index < n_params; ++index) {
            Rcpp::NumericVector col = param_columns[index];
            if (index < row.params.size()) {
                col[out_row] = row.params[index];
            }
        }
    }

    for (std::size_t index = 0; index < n_params; ++index) {
        columns.push_back(param_columns[index], result.parameter_names[index]);
    }

    return Rcpp::DataFrame(columns);
}

}  // namespace

// [[Rcpp::export(name = ".shell_rcv_d")]]
Rcpp::List r_shell_rcv_d(
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
    std::string generating_model,
    int n_draws,
    int seed,
    int threads,
    Rcpp::NumericVector lower_bounds,
    Rcpp::NumericVector upper_bounds
) {
    const std::vector<multiRL::RunTask> tasks =
        multiRL_r::make_subject_tasks(
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
            "RCV_D"
        );

    if (tasks.empty()) {
        Rcpp::stop("rcv_d requires at least one template task.");
    }

    multiRL::RecoveryControl control;
    control.n_draws = n_draws;
    control.seed = static_cast<unsigned int>(seed);
    control.threads = threads;
    control.lower_bounds = multiRL_r::as_double_vector(lower_bounds);
    control.upper_bounds = multiRL_r::as_double_vector(upper_bounds);

    const multiRL::RecoveryResult result = multiRL::shell_rcv_d(
        tasks.front(),
        control,
        generating_model
    );

    return Rcpp::List::create(
        Rcpp::_["simulation"] = wrap_recovery_simulation(result),
        Rcpp::_["truth"] = wrap_recovery_truth(result),
        Rcpp::_["metadata"] = Rcpp::List::create(
            Rcpp::_["generating_model"] = generating_model,
            Rcpp::_["n_draws"] = control.n_draws,
            Rcpp::_["seed"] = control.seed,
            Rcpp::_["threads"] = control.threads,
            Rcpp::_["parameter_names"] = result.parameter_names,
            Rcpp::_["cue_names"] = result.cue_names
        )
    );
}
