#pragma once

#include <multiRL/task_sampler.hpp>

#include <vector>

namespace multiRL {

std::vector<TaskSamplerResult> estimate_rnn(
    const std::vector<RunTask>& tasks,
    const TaskSamplerControl& control = TaskSamplerControl()
);

}  // namespace multiRL
