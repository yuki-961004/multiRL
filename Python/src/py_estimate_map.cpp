#include "py_wrapper_common.hpp"

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


void register_py_estimate_map(pybind11::module& module) {
    module.def(
        "estimate_map", &py_estimate_map,
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
