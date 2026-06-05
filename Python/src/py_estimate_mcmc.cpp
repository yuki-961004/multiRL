#include "py_wrapper_common.hpp"

#ifdef MULTIRL_HAS_STAN

pybind11::dict py_estimate_mcmc(
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
    const int warmup,
    const int samples,
    const int chains,
    const int thin,
    const double step_size,
    const double target_accept,
    const int max_tree_depth,
    const long seed,
    const std::string& algorithm,
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
        "MCMC"
    );

    multiRL::MCMCControl control;
    control.warmup = warmup;
    control.samples = samples;
    control.chains = chains;
    control.thin = thin;
    control.step_size = step_size;
    control.target_accept = target_accept;
    control.max_tree_depth = max_tree_depth;
    control.seed = seed;
    control.algorithm = algorithm;
    control.lower_bounds = lower_bounds;
    control.upper_bounds = upper_bounds;

    std::vector<multiRL::SubjectMCMCResult> results =
        multiRL::estimate_mcmc(std::vector<multiRL::RunTask>{task}, control);

    if (results.empty()) {
        throw std::runtime_error("MCMC returned no subject results.");
    }

    return py_wrap_estimate_mcmc_result(results[0], free_names, control);
}


#ifdef MULTIRL_HAS_STAN
void register_py_estimate_mcmc(pybind11::module& module) {
    module.def(
        "estimate_mcmc", &py_estimate_mcmc,
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
        pybind11::arg("warmup") = 1000,
        pybind11::arg("samples") = 1000,
        pybind11::arg("chains") = 4,
        pybind11::arg("thin") = 1,
        pybind11::arg("step_size") = 0.1,
        pybind11::arg("target_accept") = 0.8,
        pybind11::arg("max_tree_depth") = 10,
        pybind11::arg("seed") = 1004,
        pybind11::arg("algorithm") = "nuts",
        pybind11::arg("lower_bounds") = std::vector<double>(),
        pybind11::arg("upper_bounds") = std::vector<double>()
    );
}
#else
void register_py_estimate_mcmc(pybind11::module&) {}
#endif
