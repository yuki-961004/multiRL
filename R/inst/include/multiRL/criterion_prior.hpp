#pragma once

#include <multiRL/types.hpp>

namespace multiRL {

template <typename T>
T criterion_prior_value(const Params& params, const PriorGroup& priors);

double criterion_prior(const Params& params, const PriorGroup& priors);

}  // namespace multiRL
