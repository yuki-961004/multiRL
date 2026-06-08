#pragma once

#include <multiRL/modify_control.hpp>
#include <multiRL/task_builder.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace multiRL {
namespace modify_outputs {

/* ========================================================================== *
 *                         Common Output Slot Types
 * ========================================================================== */

struct OutputEstimator {
    std::string name;
    std::string backend;
    std::string algorithm;
    std::string global_algorithm;
    std::string local_algorithm;

    std::unordered_map<std::string, double> numeric_control;
    std::unordered_map<std::string, std::string> string_control;
    std::unordered_map<std::string, bool> bool_control;
};

struct OutputDiagnosticsRow {
    std::string subid;
    int status = -1;
    int n_evals = 0;
    int n_draws = 0;
    double optimum_value = 0.0;
    double accept_rate = 0.0;
    std::string result_message;
    std::string stop_reason;
};

/* ========================================================================== *
 *                         Estimator Builders
 * ========================================================================== */

// Build estimator metadata for NLopt-backed estimators (MLE / MAP).
OutputEstimator nlopt_estimator(
    const std::string& estimator_name,
    const NLoptControl& control
);

// Build estimator metadata for Stan-backed MCMC (HMC / NUTS).
OutputEstimator stan_estimator(
    const std::string& estimator_name,
    const MCMCControl& control
);

}  // namespace modify_outputs
}  // namespace multiRL
