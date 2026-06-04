#include <multiRL/estimate_rnn.hpp>

namespace multiRL {

std::vector<TaskSamplerResult> estimate_rnn(
    const std::vector<RunTask>& tasks,
    const TaskSamplerControl& control
) {
    return task_sampler(tasks, control);
}

}  // namespace multiRL
