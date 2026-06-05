#include "py_wrapper_common.hpp"

pybind11::dict py_estimate_abc(
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
    const int samples,
    const double tol,
    const std::string& method,
    const std::string& reduction,
    const int n_comp,
    const int fake_block,
    const int seed,
    const int threads,
    const int print_level,
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
        "ABC"
    );

    multiRL::ABCControl control;
    control.samples = samples;
    control.tol = tol;
    control.method = method;
    control.reduction = reduction;
    control.reduce = reduction;
    control.n_comp = n_comp;
    control.fake_block = fake_block;
    control.seed = static_cast<unsigned int>(seed);
    control.threads = threads;
    control.print_level = print_level;
    control.lower_bounds = lower_bounds;
    control.upper_bounds = upper_bounds;

    const multiRL::ABCSubjectResult result =
        multiRL::estimate_abc(task, control);
    return py_wrap_estimate_abc_result(result, control);
}


void register_py_estimate_abc(pybind11::module& module) {
    module.def(
        "estimate_abc", &py_estimate_abc,
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
        pybind11::arg("samples") = 1000,
        pybind11::arg("tol") = 0.1,
        pybind11::arg("method") = "rejection",
        pybind11::arg("reduction") = "none",
        pybind11::arg("n_comp") = 0,
        pybind11::arg("fake_block") = 0,
        pybind11::arg("seed") = 123,
        pybind11::arg("threads") = 0,
        pybind11::arg("print_level") = 1,
        pybind11::arg("lower_bounds") = std::vector<double>(),
        pybind11::arg("upper_bounds") = std::vector<double>()
    );
}
