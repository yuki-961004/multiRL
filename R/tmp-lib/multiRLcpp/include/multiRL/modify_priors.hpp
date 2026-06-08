#pragma once

#include <multiRL/types.hpp>

namespace multiRL {

PriorGroup modify_priors(
    const std::vector<std::string>& names,
    const std::vector<std::string>& types,
    const std::vector<double>& param1,
    const std::vector<double>& param2,
    const std::vector<std::string>& free_names,
    bool active
);

}  // namespace multiRL
