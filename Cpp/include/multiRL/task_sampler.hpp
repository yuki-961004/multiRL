#pragma once

#include <multiRL/types.hpp>

#include <string>
#include <vector>

namespace multiRL {

struct TaskSamplerControl {
    int n_draws = 100;
    unsigned int seed = 123;
    int threads = 0;
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
};

struct TaskSamplerRow {
    int draw = 0;
    int sequence = 0;
    int subject_index = 0;
    std::string subid;
    int block = 0;
    int trial = 0;
    std::string action;
    std::string latent;
    std::string simulation;
    std::string position;
    double reward = missing_real();
    std::vector<double> probability;
    std::vector<double> params;
};

struct TaskSamplerResult {
    std::vector<std::string> parameter_names;
    std::vector<std::string> cue_names;
    std::vector<TaskSamplerRow> rows;
    TaskSamplerControl control;
    bool generate = true;
};

TaskSamplerResult task_sampler(
    const RunTask& task,
    const TaskSamplerControl& control = TaskSamplerControl()
);

std::vector<TaskSamplerResult> task_sampler(
    const std::vector<RunTask>& tasks,
    const TaskSamplerControl& control = TaskSamplerControl()
);

}  // namespace multiRL
