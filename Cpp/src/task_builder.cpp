#include <multiRLcpp/task_builder.hpp>

namespace multiRLcpp {

RunTask task_builder(
    const Process1Input& input,
    const Process2Behrule& behrule,
    const Params& params,
    const Settings& settings,
    const PriorGroup& priors
) {
    RunTask task;
    task.input = input;
    task.behrule = behrule;
    task.params = params;
    task.settings = settings;
    task.priors = priors;
    return task;
}

}  // namespace multiRLcpp
