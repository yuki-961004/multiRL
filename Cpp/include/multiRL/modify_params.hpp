#pragma once

#include <multiRL/types.hpp>

namespace multiRL {

Params modify_params(
    const std::vector<std::string>& names,
    const std::vector<double>& values,
    const std::vector<std::string>& free_names
);

}  // namespace multiRL
