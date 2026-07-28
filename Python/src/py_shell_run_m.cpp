#include "py_wrapper_common.hpp"

pybind11::dict py_shell_run_m(
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
    const std::vector<std::string>& prior_names,
    const std::vector<std::string>& prior_types,
    const std::vector<double>& prior_param1,
    const std::vector<double>& prior_param2,
    bool prior_active,
    bool generate,
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
        generate,
        name,
        mode,
        estimate
    );

    return py_wrap_result(multiRL::shell_run_m(task), system);
}

void register_py_shell_run_m(pybind11::module& module) {
    module.def(
        "shell_run_m", &py_shell_run_m,
        pybind11::arg("object"), pybind11::arg("reward"),
        pybind11::arg("action"), pybind11::arg("block"),
        pybind11::arg("trial"), pybind11::arg("subid"),
        pybind11::arg("cue"), pybind11::arg("rsp"),
        pybind11::arg("params"), pybind11::arg("free_names"),
        pybind11::arg("system"),
        pybind11::arg("prior_names") = std::vector<std::string>(),
        pybind11::arg("prior_types") = std::vector<std::string>(),
        pybind11::arg("prior_param1") = std::vector<double>(),
        pybind11::arg("prior_param2") = std::vector<double>(),
        pybind11::arg("prior_active") = false,
        pybind11::arg("generate") = false,
        pybind11::arg("name") = "TD",
        pybind11::arg("mode") = "fitting",
        pybind11::arg("estimate") = "MLE"
    );
}

void register_py_estimate_mle(pybind11::module&);
void register_py_estimate_map(pybind11::module&);
void register_py_estimate_abc(pybind11::module&);
void register_py_estimate_rnn(pybind11::module&);
void register_py_shell_rcv_d(pybind11::module&);
void register_py_shell_rpl_e(pybind11::module&);
void register_py_estimate_mcmc(pybind11::module&);
void register_py_shell_fit_p(pybind11::module&);
void register_py_shell_run_m(pybind11::module&);

PYBIND11_MODULE(_backend, module) {
    module.doc() = "multiRL C++ backend for Python";
    register_py_estimate_mle(module);
    register_py_estimate_map(module);
    register_py_estimate_abc(module);
    register_py_estimate_rnn(module);
    register_py_shell_rcv_d(module);
    register_py_shell_rpl_e(module);
    register_py_estimate_mcmc(module);
    register_py_shell_fit_p(module);
    register_py_shell_run_m(module);
}
