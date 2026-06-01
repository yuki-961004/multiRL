#pragma once

#include <multiRL/types.hpp>

namespace multiRL {

RunTask task_builder(
    const Process1Input& input,
    const Process2Behrule& behrule,
    const Params& params,
    const Settings& settings,
    const PriorGroup& priors = PriorGroup()
);

}  // namespace multiRL
