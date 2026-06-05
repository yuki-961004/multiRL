#include "py_wrapper_common.hpp"
#include <multiRL/shell_fit_p.hpp>

pybind11::dict py_shell_fit_p(
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
    const std::string& estimator_name,
    const pybind11::dict& extra_control
) {
    std::vector<multiRL::RunTask> tasks;
    tasks.push_back(py_make_task(
        object, reward, action, block, trial, cue, rsp,
        params, free_names, system,
        prior_names, prior_types, prior_param1, prior_param2,
        prior_active, policy, name, mode, estimator_name
    ));

    multiRL::ShellFitPControl fit_control;
    fit_control.estimator = estimator_name;
    if (extra_control.contains("maxeval")) {
        fit_control.mle.nlopt.maxeval = extra_control["maxeval"].cast<int>();
    }

    multiRL::ShellFitPResult result = multiRL::shell_fit_p(tasks, fit_control);

    if (result.estimator == "mle" && !result.mle.empty()) {
        pybind11::list out;
        for (const auto& r : result.mle) {
            out.append(py_wrap_estimate_mle_result(r));
        }
        return out;
    }

    if (result.estimator == "map") {
        return py_wrap_estimate_map_result(result.map);
    }

    if (result.estimator == "abc" && !result.abc.empty()) {
        pybind11::list out;
        for (const auto& r : result.abc) {
            out.append(py_wrap_estimate_abc_result(r));
        }
        return out;
    }

    if (result.estimator == "mcmc" && !result.mcmc.empty()) {
        pybind11::list out;
        for (const auto& r : result.mcmc) {
            out.append(py_wrap_estimate_mcmc_result(r));
        }
        return out;
    }

    throw std::invalid_argument(
        "shell_fit_p: unsupported estimator " + result.estimator
    );
}

void register_py_shell_fit_p(pybind11::module& module) {
    module.def(
        "shell_fit_p", &py_shell_fit_p,
        pybind11::arg("object"), pybind11::arg("reward"),
        pybind11::arg("action"), pybind11::arg("block"),
        pybind11::arg("trial"), pybind11::arg("cue"),
        pybind11::arg("rsp"), pybind11::arg("params"),
        pybind11::arg("free_names"), pybind11::arg("system"),
        pybind11::arg("prior_names") = std::vector<std::string>(),
        pybind11::arg("prior_types") = std::vector<std::string>(),
        pybind11::arg("prior_param1") = std::vector<double>(),
        pybind11::arg("prior_param2") = std::vector<double>(),
        pybind11::arg("prior_active") = false,
        pybind11::arg("policy") = "off",
        pybind11::arg("name") = "TD",
        pybind11::arg("mode") = "fitting",
        pybind11::arg("estimator") = "mle",
        pybind11::arg("extra_control") = pybind11::dict()
    );
}
