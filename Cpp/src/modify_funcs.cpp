#include <multiRL/modify_funcs.hpp>

#include <multiRL/funcs.hpp>

#include <stdexcept>

namespace multiRL {
namespace {

void reject_custom(
    const std::string& component,
    const std::string& value
) {
    throw std::invalid_argument(
        "Custom callback functions are not implemented yet for " +
        component + ": " + value
    );
}

std::string value_or_default(
    const std::unordered_map<std::string, std::string>& funcs,
    const std::string& component,
    const std::string& default_value
) {
    const auto found = funcs.find(component);
    if (found == funcs.end() || found->second.empty()) {
        return default_value;
    }
    return found->second;
}

void require_builtin(
    const std::string& component,
    const std::string& value,
    const std::string& expected
) {
    if (value != expected) {
        reject_custom(component, value);
    }
}

}  // namespace

FunctionConfig modify_funcs(
    const std::unordered_map<std::string, std::string>& funcs
) {
    FunctionConfig out;

    out.lrng_name = value_or_default(funcs, "lrng_func", "func_alpha");
    out.prob_name = value_or_default(funcs, "prob_func", "func_beta");
    out.util_name = value_or_default(funcs, "util_func", "func_gamma");
    out.bias_name = value_or_default(funcs, "bias_func", "func_delta");
    out.expl_name = value_or_default(funcs, "expl_func", "func_epsilon");
    out.dcay_name = value_or_default(funcs, "dcay_func", "func_zeta");

    require_builtin("lrng_func", out.lrng_name, "func_alpha");
    require_builtin("prob_func", out.prob_name, "func_beta");
    require_builtin("util_func", out.util_name, "func_gamma");
    require_builtin("bias_func", out.bias_name, "func_delta");
    require_builtin("expl_func", out.expl_name, "func_epsilon");
    require_builtin("dcay_func", out.dcay_name, "func_zeta");

    out.lrng_func = func_alpha;
    out.prob_func = func_beta;
    out.util_func = func_gamma;
    out.bias_func = func_delta;
    out.expl_func = func_epsilon;
    out.dcay_func = func_zeta;

    /*
     * Future custom callbacks should probably use a context-first signature.
     * For R, the conceptual interface is lrng_func <- function(context, ...).
     * For Python, the conceptual interface is def lrng_func(context, **kwargs).
     * This file is the single place where those callbacks should be detected,
     * validated, and converted into the internal FunctionConfig.
     */

    return out;
}

}  // namespace multiRL
