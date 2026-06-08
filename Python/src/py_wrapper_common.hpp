#pragma once

// Shared C++ wrapper utilities for Python pybind11 bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <multiRL/estimate_abc.hpp>
#include <multiRL/estimate_mle.hpp>
#include <multiRL/estimate_map.hpp>
#include <multiRL/estimate_rnn.hpp>
#ifdef MULTIRL_HAS_STAN
#include <multiRL/estimate_mcmc.hpp>
#endif
#include <multiRL/modify_priors.hpp>
#include <multiRL/process_model_free.hpp>
#include <multiRL/shell_rcv_d.hpp>
#include <multiRL/shell_rpl_e.hpp>
#include <multiRL/shell_run_m.hpp>
#include <multiRL/task_builder.hpp>
#include <multiRL/task_sampler.hpp>
#include <multiRL/types.hpp>

namespace {

multiRL::Params py_params_to_cpp(
    const std::unordered_map<std::string, double>& params,
    const std::vector<std::string>& free_names
) {
    multiRL::Params out;
    out.values = params;
    out.free_names = free_names;
    return out;
}

inline pybind11::dict py_wrap_result(
    const multiRL::RunResult& result,
    const std::vector<std::string>& system
) {
    pybind11::dict value;
    for (std::size_t index = 0; index < system.size(); ++index) {
        value[pybind11::str(system[index])] =
            result.result.value[index];
    }

    pybind11::dict metric;
    metric["ACC"] = result.metric.acc;
    metric["LogL"] = result.metric.log_likelihood;
    metric["LogPr"] = result.metric.log_prior;
    metric["LogPo"] = result.metric.log_posterior;
    metric["NLL"] = result.metric.nll;
    metric["AIC"] = result.metric.aic;
    metric["BIC"] = result.metric.bic;

    pybind11::dict run_result;
    run_result["value"] = value;
    run_result["bias"] = result.result.bias;
    run_result["shown"] = result.result.shown;
    run_result["prob"] = result.result.prob;
    run_result["count"] = result.result.count;
    run_result["behave"] = result.result.behave;
    run_result["exploration"] = result.result.exploration;
    run_result["latent"] = result.result.latent;
    run_result["reward"] = result.result.reward;
    run_result["utility"] = result.result.utility;
    run_result["simulation"] = result.result.simulation;
    run_result["position"] = result.result.position;

    pybind11::dict out;
    out["metric"] = metric;
    out["result"] = run_result;
    return out;
}

inline pybind11::dict py_wrap_estimate_mle_result(
    const multiRL::EstimateMleResult& result
) {
    pybind11::dict fit;
    fit["subid"] = "1";
    for (const std::string& name : result.params.free_names) {
        fit[pybind11::str(name)] = result.params.get(name);
    }
    fit["ACC"] = result.metric.acc;
    fit["LogL"] = result.metric.log_likelihood;
    fit["LogPr"] = result.metric.log_prior;
    fit["LogPo"] = result.metric.log_posterior;
    fit["NLL"] = result.metric.nll;
    fit["AIC"] = result.metric.aic;
    fit["BIC"] = result.metric.bic;
    fit["status"] = result.status;
    fit["n_evals"] = result.n_evals;
    fit["optimum_value"] = result.optimum_value;

    pybind11::dict subject;
    subject["subid"] = "1";
    subject["status"] = result.status;
    subject["n_evals"] = result.n_evals;
    subject["optimum_value"] = result.optimum_value;
    subject["result_message"] = result.result_message;
    subject["stop_reason"] = result.stop_reason;

    pybind11::list subjects;
    subjects.append(subject);

    pybind11::dict diagnostics;
    diagnostics["subjects"] = subjects;

    pybind11::dict out;
    out["fit"] = fit;
    out["diagnostics"] = diagnostics;
    return out;
}

inline pybind11::dict py_wrap_estimate_map_result(
    const multiRL::EstimateMapResult& result
) {
    pybind11::dict out = py_wrap_estimate_mle_result(result.best);
    pybind11::dict diagnostics = out["diagnostics"].cast<pybind11::dict>();
    pybind11::dict em;
    em["iterations"] = result.iterations;
    em["delta"] = result.delta;
    em["best_log_posterior"] = result.best_log_posterior;
    em["stop_reason"] = result.stop_reason;
    diagnostics["em"] = em;
    out["diagnostics"] = diagnostics;
    return out;
}

inline pybind11::dict py_wrap_estimate_abc_result(
    const multiRL::ABCSubjectResult& result,
    const multiRL::ABCControl& control
) {
    pybind11::dict fit;
    fit["subid"] = result.subid;
    for (std::size_t index = 0;
         index < result.parameter_names.size();
         ++index) {
        if (index < result.estimates.size()) {
            fit[pybind11::str(result.parameter_names[index])] =
                result.estimates[index];
        } else {
            fit[pybind11::str(result.parameter_names[index])] =
                multiRL::missing_real();
        }
    }
    fit["status"] = result.status;

    pybind11::dict subject;
    subject["subid"] = result.subid;
    subject["status"] = result.status;
    subject["n_simulations"] = result.n_simulations;
    subject["n_accepted"] = result.n_accepted;
    subject["n_blocks_real"] = result.n_blocks_real;
    subject["n_blocks_used"] = result.n_blocks_used;
    subject["fake_block"] = result.fake_block;
    subject["n_comp_requested"] = result.n_comp_requested;
    subject["n_comp_used"] = result.n_comp_used;
    subject["reduction"] = control.reduction;
    subject["tolerance"] = control.tol;
    subject["result_message"] = result.message;

    pybind11::list subjects;
    subjects.append(subject);

    pybind11::dict control_dict;
    control_dict["samples"] = control.samples;
    control_dict["tol"] = control.tol;
    control_dict["method"] = control.method;
    control_dict["reduction"] = control.reduction;
    control_dict["n_comp"] = control.n_comp;
    control_dict["fake_block"] = control.fake_block;
    control_dict["seed"] = control.seed;
    control_dict["threads"] = control.threads;
    control_dict["print_level"] = control.print_level;

    pybind11::dict estimator;
    estimator["name"] = "ABC";
    estimator["backend"] = "abcpp";
    estimator["method"] = control.method;
    estimator["reduction"] = control.reduction;
    estimator["control"] = control_dict;

    pybind11::dict diagnostics;
    diagnostics["subjects"] = subjects;
    diagnostics["observed_summary"] = result.observed_summary;
    diagnostics["accepted_distances"] = result.accepted_distances;

    pybind11::dict out;
    out["fit"] = fit;
    out["estimator"] = estimator;
    out["diagnostics"] = diagnostics;
    return out;
}

inline pybind11::dict py_wrap_task_sampler_result(
    const multiRL::TaskSamplerResult& result
) {
    pybind11::list rows;
    for (const multiRL::TaskSamplerRow& row : result.rows) {
        pybind11::dict item;
        item["draw"] = row.draw;
        item["subid"] = row.subid;
        item["block"] = row.block;
        item["trial"] = row.trial;
        item["action"] = row.action;
        item["latent"] = row.latent;
        item["simulation"] = row.simulation;
        item["position"] = row.position;
        item["reward"] = row.reward;

        for (std::size_t index = 0; index < result.cue_names.size();
             ++index) {
            const std::string name = "prob_" + result.cue_names[index];
            if (index < row.probability.size()) {
                item[pybind11::str(name)] = row.probability[index];
            } else {
                item[pybind11::str(name)] = multiRL::missing_real();
            }
        }

        for (std::size_t index = 0; index < result.parameter_names.size();
             ++index) {
            if (index < row.params.size()) {
                item[pybind11::str(result.parameter_names[index])] =
                    row.params[index];
            } else {
                item[pybind11::str(result.parameter_names[index])] =
                    multiRL::missing_real();
            }
        }
        rows.append(item);
    }

    pybind11::dict metadata;
    metadata["n_draws"] = result.control.n_draws;
    metadata["seed"] = result.control.seed;
    metadata["threads"] = result.control.threads;
    metadata["policy"] = result.policy;
    metadata["parameter_names"] = result.parameter_names;

    pybind11::dict out;
    out["data"] = rows;
    out["metadata"] = metadata;
    return out;
}

inline pybind11::dict py_wrap_recovery_result(
    const multiRL::RecoveryResult& result
) {
    pybind11::list simulation;
    for (const multiRL::RecoverySimulationRow& row : result.simulation) {
        pybind11::dict item;
        item["generating_model"] = row.generating_model;
        item["draw"] = row.draw;
        item["subid"] = row.subid;
        item["block"] = row.block;
        item["trial"] = row.trial;
        item["action"] = row.action;
        item["latent"] = row.latent;
        item["simulation"] = row.simulation;
        item["position"] = row.position;
        item["sampled_reward"] = row.reward;

        for (std::size_t index = 0; index < row.object.size(); ++index) {
            item[pybind11::str("object_" + std::to_string(index + 1))] =
                row.object[index];
        }
        for (std::size_t index = 0; index < row.reward_table.size(); ++index) {
            item[pybind11::str("reward_" + std::to_string(index + 1))] =
                row.reward_table[index];
        }
        for (std::size_t index = 0; index < result.cue_names.size(); ++index) {
            const std::string name = "prob_" + result.cue_names[index];
            if (index < row.probability.size()) {
                item[pybind11::str(name)] = row.probability[index];
            } else {
                item[pybind11::str(name)] = multiRL::missing_real();
            }
        }
        simulation.append(item);
    }

    pybind11::list truth;
    for (const multiRL::RecoveryTruthRow& row : result.truth) {
        pybind11::dict item;
        item["generating_model"] = row.generating_model;
        item["draw"] = row.draw;
        item["subid"] = row.subid;
        for (std::size_t index = 0; index < result.parameter_names.size();
             ++index) {
            if (index < row.params.size()) {
                item[pybind11::str(result.parameter_names[index])] =
                    row.params[index];
            } else {
                item[pybind11::str(result.parameter_names[index])] =
                    multiRL::missing_real();
            }
        }
        truth.append(item);
    }

    pybind11::dict metadata;
    metadata["n_draws"] = result.control.n_draws;
    metadata["seed"] = result.control.seed;
    metadata["threads"] = result.control.threads;
    metadata["parameter_names"] = result.parameter_names;
    metadata["cue_names"] = result.cue_names;

    pybind11::dict out;
    out["simulation"] = simulation;
    out["truth"] = truth;
    out["metadata"] = metadata;
    return out;
}

inline pybind11::dict py_wrap_replay_result(
    const multiRL::ReplayResult& result
) {
    pybind11::list replay;
    for (const multiRL::ReplayRow& row : result.replay) {
        pybind11::dict item;
        item["source"] = row.source;
        item["model"] = row.model;
        item["model_id"] = row.model_id;
        item["subid"] = row.subid;
        item["block"] = row.block;
        item["trial"] = row.trial;
        item["action"] = row.action;
        item["reward"] = row.reward;
        replay.append(item);
    }

    pybind11::list plot_data;
    for (const multiRL::ReplayPlotRow& row : result.plot_data) {
        pybind11::dict item;
        item["source"] = row.source;
        item["model"] = row.model;
        item["model_id"] = row.model_id;
        item["subid"] = row.subid;
        item["block"] = row.block;
        item["action"] = row.action;
        item["count"] = row.count;
        item["n"] = row.n;
        item["ratio"] = row.ratio;
        plot_data.append(item);
    }

    pybind11::dict diagnostics;
    diagnostics["n_models"] = result.n_models;
    diagnostics["n_subjects"] = result.n_subjects;
    diagnostics["policy"] = result.policy;
    diagnostics["replay_success"] = result.replay_success;

    pybind11::dict out;
    out["replay"] = replay;
    out["plot_data"] = plot_data;
    out["diagnostics"] = diagnostics;
    return out;
}

std::vector<multiRL::ReplayFit> py_replay_fits(
    const pybind11::list& fit_rows,
    const std::vector<std::string>& parameter_names
) {
    std::vector<multiRL::ReplayFit> out;
    out.reserve(static_cast<std::size_t>(pybind11::len(fit_rows)));

    for (pybind11::handle handle : fit_rows) {
        pybind11::dict row = pybind11::cast<pybind11::dict>(handle);
        multiRL::ReplayFit fit;
        fit.model = pybind11::cast<std::string>(row["model"]);
        fit.model_id = pybind11::cast<std::string>(row["model_id"]);
        fit.subid = pybind11::cast<std::string>(row["subid"]);

        for (const std::string& name : parameter_names) {
            if (row.contains(pybind11::str(name))) {
                pybind11::object value = row[pybind11::str(name)];
                if (!value.is_none()) {
                    fit.params[name] = pybind11::cast<double>(value);
                }
            }
        }

        out.push_back(fit);
    }

    return out;
}

#ifdef MULTIRL_HAS_STAN

inline pybind11::dict py_wrap_estimate_mcmc_result(
    const multiRL::SubjectMCMCResult& result,
    const std::vector<std::string>& free_names,
    const multiRL::MCMCControl& control
) {
    pybind11::dict fit;
    fit["subid"] = result.subid;
    for (std::size_t index = 0; index < free_names.size(); ++index) {
        if (!result.best_params.empty() &&
            index < result.best_params[0].size()) {
            fit[pybind11::str(free_names[index])] =
                result.best_params[0][index];
        } else {
            fit[pybind11::str(free_names[index])] =
                multiRL::missing_real();
        }
    }
    fit["LogL"] = result.logL;
    fit["LogPr"] = result.logPrior;
    fit["LogPo"] = result.logPost;
    fit["AIC"] = result.aic;
    fit["BIC"] = result.bic;
    fit["status"] = result.status;
    fit["n_evals"] = result.n_evals;
    fit["n_draws"] = result.n_draws;

    pybind11::dict subject;
    subject["subid"] = result.subid;
    subject["status"] = result.status;
    subject["n_evals"] = result.n_evals;
    subject["n_draws"] = result.n_draws;
    subject["result_message"] = result.result_message;
    subject["stop_reason"] = result.stop_reason;

    pybind11::list subjects;
    subjects.append(subject);

    pybind11::dict estimator;
    estimator["name"] = "MCMC";
    estimator["backend"] = "stan";
    estimator["algorithm"] =
        (control.algorithm == "nuts") ? "NUTS" : "Static HMC";
    estimator["global_algorithm"] = estimator["algorithm"];
    estimator["local_algorithm"] = "";

    pybind11::dict diagnostics;
    diagnostics["subjects"] = subjects;

    pybind11::dict out;
    out["fit"] = fit;
    out["estimator"] = estimator;
    out["diagnostics"] = diagnostics;
    return out;
}

#endif

}  // namespace

inline multiRL::RunTask py_make_task(
    const multiRL::StringMatrix& object,
    const multiRL::DoubleMatrix& reward,
    const std::vector<std::string>& action,
    const std::vector<int>& block,
    const std::vector<int>& trial,
    const std::vector<std::string>& cue,
    const std::vector<std::string>& rsp,
    const std::unordered_map<std::string, double>& params,
    const std::vector<std::string>& free_names,
    const std::vector<std::string>& system,
    const std::vector<std::string>& prior_names,
    const std::vector<std::string>& prior_types,
    const std::vector<double>& prior_param1,
    const std::vector<double>& prior_param2,
    bool prior_active,
    const std::string& policy,
    const std::string& name,
    const std::string& mode,
    const std::string& estimate
) {
    multiRL::Process1Input input = multiRL::process_1_input(
        object,
        reward,
        action,
        block,
        trial,
        multiRL::StringMatrix(action.size()),
        multiRL::StringMatrix(action.size())
    );

    multiRL::Process2Behrule behrule = multiRL::process_2_behrule(
        cue,
        rsp
    );

    multiRL::Settings settings;
    settings.policy = policy;
    settings.name = name;
    settings.mode = mode;
    settings.estimate = estimate;
    settings.system = system;

    multiRL::RunTask task = multiRL::task_builder(
        input,
        behrule,
        py_params_to_cpp(params, free_names),
        settings
    );
    task.priors = multiRL::modify_priors(
        prior_names,
        prior_types,
        prior_param1,
        prior_param2,
        free_names,
        prior_active
    );

    return task;
}
