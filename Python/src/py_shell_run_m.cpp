#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <unordered_map>
#include <vector>

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

}  // namespace

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

    multiRL::RunResult result = multiRL::shell_run_m(task);
    return py_wrap_result(result, system);
}

PYBIND11_MODULE(_shell_run_m, module) {
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
}
