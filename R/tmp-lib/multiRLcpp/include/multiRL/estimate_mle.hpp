#pragma once

#include <multiRL/types.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace multiRL {

struct NLoptControl {
    std::string algorithm = "GN_MLSL";
    std::string local_algorithm = "LN_BOBYQA";
    double local_xtol_rel = 1e-8;
    double xtol_rel = 1e-6;
    double ftol_rel = 0.0;
    double ftol_abs = 0.0;
    double xtol_abs = 0.0;
    double maxtime = 0.0;
    double stopval = 0.0;
    int maxeval = 10000;
    int population = 0;
    double initial_step = 0.0;
    long seed = 1004;
    int print_level = 1;
    int threads = 0;
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
};

struct MLEControl {
    NLoptControl nlopt;
};

struct EstimateMleResult {
    Params params;
    CriterionResult metric;
    int status = -1;
    int n_evals = 0;
    double optimum_value = missing_real();
    std::string result_message;
    std::string stop_reason;
};

EstimateMleResult estimate_mle(
    const RunTask& task,
    const MLEControl& control = MLEControl()
);

std::vector<EstimateMleResult> estimate_mle(
    const std::vector<RunTask>& tasks,
    const MLEControl& control = MLEControl()
);

EstimateMleResult estimate_mle(
    const RunTask& task,
    const NLoptControl& control
);

}  // namespace multiRL
