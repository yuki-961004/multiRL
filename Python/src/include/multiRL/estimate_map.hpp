#pragma once

#include <multiRL/estimate_mle.hpp>
#include <multiRL/modify_control.hpp>

namespace multiRL {

struct EstimateMapResult {
    EstimateMleResult best;
    std::vector<EstimateMleResult> subjects;
    PriorGroup priors;
    int iterations = 0;
    double delta = 0.0;
    double best_log_posterior = missing_real();
    std::string stop_reason;
};

EstimateMapResult estimate_map(
    const RunTask& task,
    const MAPControl& control = MAPControl()
);

EstimateMapResult estimate_map(
    const std::vector<RunTask>& tasks,
    const MAPControl& control = MAPControl()
);

}  // namespace multiRL
