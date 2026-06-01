#include <multiRL/modify_priors.hpp>

#include <stdexcept>
#include <unordered_set>

namespace multiRL {

namespace {

PriorType prior_type_from_string(const std::string& type) {
    if (type == "normal" || type == "norm") {
        return PriorType::NORMAL;
    }
    if (type == "uniform" || type == "unif") {
        return PriorType::UNIFORM;
    }
    if (type == "lognormal" || type == "lnorm") {
        return PriorType::LOGNORMAL;
    }
    if (type == "cauchy") {
        return PriorType::CAUCHY;
    }
    if (type == "beta") {
        return PriorType::BETA;
    }
    if (type == "exponential" || type == "exp") {
        return PriorType::EXPONENTIAL;
    }
    if (type == "none") {
        return PriorType::NONE;
    }

    throw std::invalid_argument("Unknown prior type.");
}

}  // namespace

/* ========================================================================== *
 *                         Prior Standardization                              *
 * ========================================================================== */

PriorGroup modify_priors(
    const std::vector<std::string>& names,
    const std::vector<std::string>& types,
    const std::vector<double>& param1,
    const std::vector<double>& param2,
    const std::vector<std::string>& free_names,
    bool active
) {
    PriorGroup out;
    out.active = active;

    if (!active) {
        return out;
    }

    if (names.size() != types.size() ||
        names.size() != param1.size() ||
        names.size() != param2.size()) {
        throw std::invalid_argument("Prior specification sizes do not match.");
    }

    std::unordered_set<std::string> free_set;
    for (const std::string& name : free_names) {
        free_set.insert(name);
    }

    for (std::size_t index = 0; index < names.size(); ++index) {
        if (free_set.find(names[index]) == free_set.end()) {
            throw std::invalid_argument(
                "Prior names must match free parameter names."
            );
        }

        PriorSpec spec;
        spec.type = prior_type_from_string(types[index]);
        spec.param1 = param1[index];
        spec.param2 = param2[index];
        out.specs[names[index]] = spec;
    }

    return out;
}

}  // namespace multiRL
