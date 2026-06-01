#pragma once

#include <multiRL/types.hpp>

namespace multiRL {

CriterionResult criterion_posterior(
    const RunTask& task,
    const Process3Record& output
);

}  // namespace multiRL
