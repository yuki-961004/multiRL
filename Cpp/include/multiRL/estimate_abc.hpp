#pragma once

#include <multiRL/modify_control.hpp>
#include <multiRL/types.hpp>

#include <string>
#include <vector>

namespace multiRL {

struct ABCSummaryStats {
    double min = missing_real();
    double q_lower = missing_real();
    double median = missing_real();
    double mean = missing_real();
    double mode = missing_real();
    double q_upper = missing_real();
    double max = missing_real();
    double sd = missing_real();
};

struct ABCSubjectResult {
    std::string subid = "1";
    std::vector<std::string> parameter_names;
    std::vector<ABCSummaryStats> parameter_summary;
    std::vector<double> estimates;
    std::vector<double> observed_summary;
    std::vector<double> accepted_distances;
    std::vector<std::size_t> accepted_indices;
    std::vector<double> accepted_weights;
    int n_simulations = 0;
    int n_accepted = 0;
    int n_comp_used = 0;
    int status = -1;
    std::string message;
};

ABCSubjectResult estimate_abc(
    const RunTask& task,
    const ABCControl& control = ABCControl()
);

std::vector<ABCSubjectResult> estimate_abc(
    const std::vector<RunTask>& tasks,
    const ABCControl& control = ABCControl()
);

}  // namespace multiRL
