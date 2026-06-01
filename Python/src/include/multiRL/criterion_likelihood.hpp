#pragma once

#include <multiRL/types.hpp>

namespace multiRL {

template <typename T>
CriterionValue<T> criterion_likelihood_value(
    const RunTask& task,
    const Process3Record& output
);

CriterionResult criterion_likelihood(
    const RunTask& task,
    const Process3Record& output
);

}  // namespace multiRL
