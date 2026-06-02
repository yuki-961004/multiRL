#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <unordered_map>
#include <vector>

#include <multiRL/estimate_mle.hpp>
#include <multiRL/estimate_map.hpp>
#include <multiRL/modify_priors.hpp>
#include <multiRL/process_MDP_free.hpp>
#include <multiRL/shell_run_m.hpp>
#include <multiRL/task_builder.hpp>
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

pybind11::dict py_wrap_result(
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

pybind11::dict py_wrap_estimate_mle_result(
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

pybind11::dict py_wrap_estimate_map_result(
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

}  // namespace

multiRL::RunTask py_make_task(
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

pybind11::dict py_shell_run_m(
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
    multiRL::RunTask task = py_make_task(
        object,
        reward,
        action,
        block,
        trial,
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
    return py_wrap_result(result, system);
}

pybind11::dict py_estimate_mle(
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
    const int maxeval,
    const std::string& algorithm,
    const std::string& local_algorithm,
    const double xtol_rel,
    const double local_xtol_rel,
    const int seed,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds
) {
    multiRL::RunTask task = py_make_task(
        object,
        reward,
        action,
        block,
        trial,
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
        "MLE"
    );

    multiRL::MLEControl control;
    control.nlopt.maxeval = maxeval;
    control.nlopt.algorithm = algorithm;
    control.nlopt.local_algorithm = local_algorithm;
    control.nlopt.xtol_rel = xtol_rel;
    control.nlopt.local_xtol_rel = local_xtol_rel;
    control.nlopt.seed = seed;
    control.nlopt.lower_bounds = lower_bounds;
    control.nlopt.upper_bounds = upper_bounds;

    multiRL::EstimateMleResult result = multiRL::estimate_mle(task, control);
    return py_wrap_estimate_mle_result(result);
}

pybind11::dict py_estimate_map(
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
    const int mle_maxeval,
    const int map_maxiter,
    const double map_tol,
    const int map_patience,
    const std::string& algorithm,
    const std::string& local_algorithm,
    const double xtol_rel,
    const double local_xtol_rel,
    const int seed,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds
) {
    multiRL::RunTask task = py_make_task(
        object,
        reward,
        action,
        block,
        trial,
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
    control.mle.nlopt.lower_bounds = lower_bounds;
    control.mle.nlopt.upper_bounds = upper_bounds;
    control.map_maxiter = map_maxiter;
    control.map_tol = map_tol;
    control.map_patience = map_patience;

    multiRL::EstimateMapResult result = multiRL::estimate_map(task, control);
    return py_wrap_estimate_map_result(result);
}

PYBIND11_MODULE(_backend, module) {
    module.doc() = "multiRL shell_run_m wrapper";
    module.def(
        "shell_run_m",
        &py_shell_run_m,
        pybind11::arg("object"),
        pybind11::arg("reward"),
        pybind11::arg("action"),
        pybind11::arg("block"),
        pybind11::arg("trial"),
        pybind11::arg("cue"),
        pybind11::arg("rsp"),
        pybind11::arg("params"),
        pybind11::arg("free_names"),
        pybind11::arg("system"),
        pybind11::arg("prior_names") = std::vector<std::string>(),
        pybind11::arg("prior_types") = std::vector<std::string>(),
        pybind11::arg("prior_param1") = std::vector<double>(),
        pybind11::arg("prior_param2") = std::vector<double>(),
        pybind11::arg("prior_active") = false,
        pybind11::arg("policy") = "off",
        pybind11::arg("name") = "TD",
        pybind11::arg("mode") = "LBI",
        pybind11::arg("estimate") = "MLE"
    );
    module.def(
        "estimate_mle",
        &py_estimate_mle,
        pybind11::arg("object"),
        pybind11::arg("reward"),
        pybind11::arg("action"),
        pybind11::arg("block"),
        pybind11::arg("trial"),
        pybind11::arg("cue"),
        pybind11::arg("rsp"),
        pybind11::arg("params"),
        pybind11::arg("free_names"),
        pybind11::arg("system"),
        pybind11::arg("prior_names") = std::vector<std::string>(),
        pybind11::arg("prior_types") = std::vector<std::string>(),
        pybind11::arg("prior_param1") = std::vector<double>(),
        pybind11::arg("prior_param2") = std::vector<double>(),
        pybind11::arg("prior_active") = false,
        pybind11::arg("policy") = "off",
        pybind11::arg("name") = "TD",
        pybind11::arg("mode") = "fitting",
        pybind11::arg("maxeval") = 10000,
        pybind11::arg("algorithm") = "GN_MLSL",
        pybind11::arg("local_algorithm") = "LN_BOBYQA",
        pybind11::arg("xtol_rel") = 1e-6,
        pybind11::arg("local_xtol_rel") = 1e-8,
        pybind11::arg("seed") = 1004,
        pybind11::arg("lower_bounds") = std::vector<double>(),
        pybind11::arg("upper_bounds") = std::vector<double>()
    );
    module.def(
        "estimate_map",
        &py_estimate_map,
        pybind11::arg("object"),
        pybind11::arg("reward"),
        pybind11::arg("action"),
        pybind11::arg("block"),
        pybind11::arg("trial"),
        pybind11::arg("cue"),
        pybind11::arg("rsp"),
        pybind11::arg("params"),
        pybind11::arg("free_names"),
        pybind11::arg("system"),
        pybind11::arg("prior_names") = std::vector<std::string>(),
        pybind11::arg("prior_types") = std::vector<std::string>(),
        pybind11::arg("prior_param1") = std::vector<double>(),
        pybind11::arg("prior_param2") = std::vector<double>(),
        pybind11::arg("prior_active") = false,
        pybind11::arg("policy") = "off",
        pybind11::arg("name") = "TD",
        pybind11::arg("mode") = "fitting",
        pybind11::arg("mle_maxeval") = 10000,
        pybind11::arg("map_maxiter") = 10,
        pybind11::arg("map_tol") = 1e-3,
        pybind11::arg("map_patience") = 10,
        pybind11::arg("algorithm") = "GN_MLSL",
        pybind11::arg("local_algorithm") = "LN_BOBYQA",
        pybind11::arg("xtol_rel") = 1e-6,
        pybind11::arg("local_xtol_rel") = 1e-8,
        pybind11::arg("seed") = 1004,
        pybind11::arg("lower_bounds") = std::vector<double>(),
        pybind11::arg("upper_bounds") = std::vector<double>()
    );
}
