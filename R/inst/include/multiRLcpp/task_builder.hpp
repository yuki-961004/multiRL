#pragma once

#include <multiRLcpp/types.hpp>

namespace multiRLcpp {

RunTask task_builder(
    const Process1Input& input,
    const Process2Behrule& behrule,
    const Params& params,
    const Settings& settings,
    const PriorGroup& priors = PriorGroup()
);

}  // namespace multiRLcpp
