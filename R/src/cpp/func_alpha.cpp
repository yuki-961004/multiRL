#include <multiRL/funcs.hpp>

#include <cmath>
#include <stdexcept>

namespace multiRL {

double func_alpha(
    const TrialContext& context,
    const Params& params
) {
    (void) context.reward;

    const double q0 = params.get("Q0");
    const bool has_alpha = params.has("alpha");
    const bool has_alpha_n = params.has("alphaN");
    const bool has_alpha_p = params.has("alphaP");
    const bool is_fp = context.is_fp;
    const double qvalue = context.qi;
    const double utility = context.utility;
    const std::string& system = context.system;

    if (std::isnan(q0) && is_fp) {
        return utility;
    }

    if (system == "RL" && has_alpha && !has_alpha_n && !has_alpha_p) {
        const double alpha = params.get("alpha");
        return qvalue + alpha * (utility - qvalue);
    }

    if (system == "RL" && !has_alpha && has_alpha_n && has_alpha_p) {
        if (utility < qvalue) {
            const double alpha_n = params.get("alphaN");
            return qvalue + alpha_n * (utility - qvalue);
        }

        const double alpha_p = params.get("alphaP");
        return qvalue + alpha_p * (utility - qvalue);
    }

    if (system == "WM") {
        return context.reward;
    }

    throw std::invalid_argument("Unknown model in func_alpha.");
}

}  // namespace multiRL
