#include <multiRLcpp/modify_params.hpp>

namespace multiRLcpp {

bool Params::has(const std::string& name) const {
    return values.find(name) != values.end();
}

double Params::get(const std::string& name) const {
    auto it = values.find(name);
    if (it == values.end()) {
        return missing_real();
    }
    return it->second;
}

Params modify_params(
    const std::vector<std::string>& names,
    const std::vector<double>& values,
    const std::vector<std::string>& free_names
) {
    Params out;

    for (std::size_t index = 0; index < names.size(); ++index) {
        if (out.values.find(names[index]) == out.values.end()) {
            out.values[names[index]] = values[index];
        }
    }

    out.free_names = free_names;
    return out;
}

}  // namespace multiRLcpp
