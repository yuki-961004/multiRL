#pragma once

#include <multiRL/types.hpp>

#include <string>
#include <unordered_map>

namespace multiRL {

FunctionConfig modify_funcs(
    const std::unordered_map<std::string, std::string>& funcs = {}
);

}  // namespace multiRL
