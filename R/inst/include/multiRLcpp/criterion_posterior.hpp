#pragma once

#include <multiRLcpp/types.hpp>

namespace multiRLcpp {

CriterionResult criterion_posterior(
    const RunTask& task,
    const Process3Record& output
);

}  // namespace multiRLcpp
