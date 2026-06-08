#include "py_wrapper_common.hpp"

namespace {

pybind11::dict py_wrap_estimate_rnn_results(
    const std::vector<multiRL::EstimateRnnSubjectResult>& results,
    const multiRL::RNNControl& control
) {
    pybind11::list fit_rows;
    pybind11::list subject_rows;

    for (const multiRL::EstimateRnnSubjectResult& result : results) {
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
        fit["n_draws"] = result.n_draws;
        fit["loss"] = result.loss;
        fit["message"] = result.message;
        fit_rows.append(fit);

        pybind11::dict subject;
        subject["subid"] = result.subid;
        subject["status"] = result.status;
        subject["n_draws"] = result.n_draws;
        subject["n_trials"] = result.n_trials;
        subject["n_features"] = result.n_features;
        subject["epochs"] = result.epochs;
        subject["loss"] = result.loss;
        subject["architecture"] = result.architecture;
        subject["message"] = result.message;
        subject_rows.append(subject);
    }

    pybind11::dict estimator;
    estimator["name"] = "RNN";
    estimator["backend"] = "torch";
    estimator["architecture"] = control.model_type;

    pybind11::dict diagnostics;
    diagnostics["subjects"] = subject_rows;

    pybind11::dict out;
    out["fit"] = fit_rows;
    out["estimator"] = estimator;
    out["diagnostics"] = diagnostics;
    return out;
}

}  // namespace

pybind11::dict py_estimate_rnn_data(
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
    const int n_draws,
    const int seed,
    const int threads,
    const int epochs,
    const int batch_size,
    const int units,
    const int layers,
    const double dropout,
    const double learning_rate,
    const std::string& model_type,
    const int verbose,
    const std::string& device,
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
        "RNN"
    );

    multiRL::RNNControl control;
    control.n_draws = n_draws;
    control.sample = n_draws;
    control.seed = seed;
    control.threads = threads;
    control.epoch = epochs;
    control.epochs = epochs;
    control.batch_size = batch_size;
    control.units = units;
    control.layers = layers;
    control.dropout = dropout;
    control.learning_rate = learning_rate;
    control.model_type = model_type;
    control.verbose = verbose;
    control.device = device;
    control.lower_bounds = lower_bounds;
    control.upper_bounds = upper_bounds;

    const std::vector<multiRL::EstimateRnnSubjectResult> result =
        multiRL::estimate_rnn(std::vector<multiRL::RunTask>{task}, control);
    return py_wrap_estimate_rnn_results(result, control);
}


void register_py_estimate_rnn(pybind11::module& module) {
    module.def(
        "estimate_rnn_data", &py_estimate_rnn_data,
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
        pybind11::arg("n_draws") = 100,
        pybind11::arg("seed") = 123,
        pybind11::arg("threads") = 0,
        pybind11::arg("epochs") = 20,
        pybind11::arg("batch_size") = 32,
        pybind11::arg("units") = 32,
        pybind11::arg("layers") = 1,
        pybind11::arg("dropout") = 0.0,
        pybind11::arg("learning_rate") = 0.001,
        pybind11::arg("model_type") = "gru",
        pybind11::arg("verbose") = 0,
        pybind11::arg("device") = "cpu",
        pybind11::arg("lower_bounds") = std::vector<double>(),
        pybind11::arg("upper_bounds") = std::vector<double>()
    );
}
