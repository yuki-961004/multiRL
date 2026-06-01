#pragma once

#include <multiRLcpp/types.hpp>

namespace multiRLcpp {

template <typename T>
T criterion_prior_value(const Params& params, const PriorGroup& priors);

double criterion_prior(const Params& params, const PriorGroup& priors);

}  // namespace multiRLcpp
