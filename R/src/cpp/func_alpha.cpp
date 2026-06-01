#include <multiRLcpp/funcs.hpp>

#include <cmath>
#include <stdexcept>

namespace multiRLcpp {

double func_alpha(
    const TrialContext& context,
    bool is_fp,
    double qvalue,
    double reward,
    double utility,
    const std::string& system,
    const Params& params
) {
    (void) context;
    (void) reward;

    const double q0 = params.get("Q0");
    const bool has_alpha = params.has("alpha");
    const bool has_alpha_n = params.has("alphaN");
    const bool has_alpha_p = params.has("alphaP");

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
        return reward;
    }

    throw std::invalid_argument("Unknown model in func_alpha.");
}

}  // namespace multiRLcpp
