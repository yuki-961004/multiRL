#include <Rcpp.h>

#include <multiRL/shell_rpl_e.hpp>

#include "r_wrapper_common.hpp"

#include <string>
#include <unordered_set>
#include <vector>

namespace {

std::vector<multiRL::ReplayFit> as_replay_fits(
    const Rcpp::DataFrame& fit,
    const Rcpp::CharacterVector& parameter_names
) {
    const Rcpp::CharacterVector fit_names = fit.names();
    const std::unordered_set<std::string> columns(
        fit_names.begin(),
        fit_names.end()
    );
    const R_xlen_t n_rows = fit.nrows();
    std::vector<multiRL::ReplayFit> out;
    out.reserve(static_cast<std::size_t>(n_rows));

    Rcpp::CharacterVector subid = fit["subid"];
    Rcpp::CharacterVector model = columns.count("model") > 0
        ? Rcpp::CharacterVector(fit["model"])
        : Rcpp::CharacterVector(n_rows, "model");
    Rcpp::CharacterVector model_id = columns.count("model_id") > 0
        ? Rcpp::CharacterVector(fit["model_id"])
        : model;

    for (R_xlen_t row = 0; row < n_rows; ++row) {
        multiRL::ReplayFit item;
        item.subid = Rcpp::as<std::string>(subid[row]);
        item.model = Rcpp::as<std::string>(model[row]);
        item.model_id = Rcpp::as<std::string>(model_id[row]);

        for (R_xlen_t index = 0; index < parameter_names.size(); ++index) {
            const std::string name = Rcpp::as<std::string>(
                parameter_names[index]
            );
            if (columns.count(name) == 0) {
                continue;
            }

            Rcpp::NumericVector values = fit[name];
            if (!Rcpp::NumericVector::is_na(values[row])) {
                item.params[name] = values[row];
            }
        }

        out.push_back(item);
    }

    return out;
}

Rcpp::DataFrame wrap_replay(const std::vector<multiRL::ReplayRow>& rows) {
    const std::size_t n = rows.size();
    Rcpp::CharacterVector source(n);
    Rcpp::CharacterVector model(n);
    Rcpp::CharacterVector model_id(n);
    Rcpp::CharacterVector subid(n);
    Rcpp::IntegerVector block(n);
    Rcpp::IntegerVector trial(n);
    Rcpp::CharacterVector action(n);
    Rcpp::NumericVector reward(n);

    for (std::size_t row = 0; row < n; ++row) {
        source[row] = rows[row].source;
        model[row] = rows[row].model;
        model_id[row] = rows[row].model_id;
        subid[row] = rows[row].subid;
        block[row] = rows[row].block;
        trial[row] = rows[row].trial;
        action[row] = rows[row].action;
        reward[row] = rows[row].reward;
    }

    return Rcpp::DataFrame::create(
        Rcpp::_["source"] = source,
        Rcpp::_["model"] = model,
        Rcpp::_["model_id"] = model_id,
        Rcpp::_["subid"] = subid,
        Rcpp::_["block"] = block,
        Rcpp::_["trial"] = trial,
        Rcpp::_["action"] = action,
        Rcpp::_["reward"] = reward
    );
}

Rcpp::DataFrame wrap_plot_data(
    const std::vector<multiRL::ReplayPlotRow>& rows
) {
    const std::size_t n = rows.size();
    Rcpp::CharacterVector source(n);
    Rcpp::CharacterVector model(n);
    Rcpp::CharacterVector model_id(n);
    Rcpp::CharacterVector subid(n);
    Rcpp::IntegerVector block(n);
    Rcpp::CharacterVector action(n);
    Rcpp::IntegerVector count(n);
    Rcpp::IntegerVector total(n);
    Rcpp::NumericVector ratio(n);

    for (std::size_t row = 0; row < n; ++row) {
        source[row] = rows[row].source;
        model[row] = rows[row].model;
        model_id[row] = rows[row].model_id;
        subid[row] = rows[row].subid;
        block[row] = rows[row].block;
        action[row] = rows[row].action;
        count[row] = rows[row].count;
        total[row] = rows[row].n;
        ratio[row] = rows[row].ratio;
    }

    return Rcpp::DataFrame::create(
        Rcpp::_["source"] = source,
        Rcpp::_["model"] = model,
        Rcpp::_["model_id"] = model_id,
        Rcpp::_["subid"] = subid,
        Rcpp::_["block"] = block,
        Rcpp::_["action"] = action,
        Rcpp::_["count"] = count,
        Rcpp::_["n"] = total,
        Rcpp::_["ratio"] = ratio
    );
}

}  // namespace

// [[Rcpp::export(name = ".shell_rpl_e")]]
Rcpp::List r_shell_rpl_e(
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
    bool generate,
    std::string name,
    std::string mode,
    Rcpp::DataFrame fit,
    Rcpp::CharacterVector parameter_names
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
        generate,
        name,
        mode,
        "RPL_E"
    );

    const multiRL::ReplayResult result = multiRL::shell_rpl_e(
        tasks,
        as_replay_fits(fit, parameter_names)
    );

    return Rcpp::List::create(
        Rcpp::_["replay"] = wrap_replay(result.replay),
        Rcpp::_["plot_data"] = wrap_plot_data(result.plot_data),
        Rcpp::_["diagnostics"] = Rcpp::List::create(
            Rcpp::_["n_models"] = result.n_models,
            Rcpp::_["n_subjects"] = result.n_subjects,
            Rcpp::_["generate"] = result.generate,
            Rcpp::_["replay_success"] = result.replay_success
        )
    );
}
