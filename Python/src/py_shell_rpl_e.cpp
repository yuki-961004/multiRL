#include "py_wrapper_common.hpp"

pybind11::dict py_shell_rpl_e(
    const multiRL::StringMatrix& object,
    const multiRL::DoubleMatrix& reward,
    const std::vector<std::string>& action,
    const std::vector<int>& block,
    const std::vector<int>& trial,
    const std::vector<std::string>& subid,
    const std::vector<std::string>& cue,
    const std::vector<std::string>& rsp,
    const std::unordered_map<std::string, double>& params,
    const std::vector<std::string>& free_names,
    const std::vector<std::string>& system,
    const pybind11::list& fit_rows,
    const std::vector<std::string>& parameter_names,
    const std::vector<std::string>& prior_names,
    const std::vector<std::string>& prior_types,
    const std::vector<double>& prior_param1,
    const std::vector<double>& prior_param2,
    bool prior_active,
    const std::string& policy,
    const std::string& name,
    const std::string& mode
) {
    multiRL::RunTask task = py_make_task(
        object,
        reward,
        action,
        block,
        trial,
        subid,
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
        "RPL_E"
    );

    std::vector<multiRL::RunTask> tasks =
        multiRL::split_task_by_subject(task);
    const multiRL::ReplayResult result = multiRL::shell_rpl_e(
        tasks,
        py_replay_fits(fit_rows, parameter_names)
    );
    return py_wrap_replay_result(result);
}


void register_py_shell_rpl_e(pybind11::module& module) {
    module.def(
        "shell_rpl_e", &py_shell_rpl_e,
        pybind11::arg("object"), pybind11::arg("reward"),
        pybind11::arg("action"), pybind11::arg("block"),
        pybind11::arg("trial"), pybind11::arg("subid"),
        pybind11::arg("cue"), pybind11::arg("rsp"),
        pybind11::arg("params"), pybind11::arg("free_names"),
        pybind11::arg("system"), pybind11::arg("fit_rows"),
        pybind11::arg("parameter_names"),
        pybind11::arg("prior_names") = std::vector<std::string>(),
        pybind11::arg("prior_types") = std::vector<std::string>(),
        pybind11::arg("prior_param1") = std::vector<double>(),
        pybind11::arg("prior_param2") = std::vector<double>(),
        pybind11::arg("prior_active") = false,
        pybind11::arg("policy") = "off",
        pybind11::arg("name") = "TD",
        pybind11::arg("mode") = "fitting"
    );
}
