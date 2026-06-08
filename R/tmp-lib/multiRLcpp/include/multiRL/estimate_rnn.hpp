#pragma once

#include <multiRL/modify_control.hpp>
#include <multiRL/task_sampler.hpp>

#include <vector>

namespace multiRL {

struct EstimateRnnSubjectResult {
    std::string subid = "1";
    std::vector<std::string> parameter_names;
    std::vector<double> estimates;
    int status = -1;
    int n_draws = 0;
    int n_trials = 0;
    int n_features = 0;
    int epochs = 0;
    double loss = missing_real();
    std::string backend = "torch";
    std::string architecture = "gru";
    std::string message;
};

std::vector<TaskSamplerResult> estimate_rnn(
    const std::vector<RunTask>& tasks,
    const TaskSamplerControl& control = TaskSamplerControl()
);

std::vector<EstimateRnnSubjectResult> estimate_rnn(
    const std::vector<RunTask>& tasks,
    const RNNControl& control = RNNControl()
);

}  // namespace multiRL
