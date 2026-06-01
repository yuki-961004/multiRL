#pragma once

#include <multiRL/types.hpp>

namespace multiRL {

double func_alpha(
    const TrialContext& context,
    bool is_fp,
    double qvalue,
    double reward,
    double utility,
    const std::string& system,
    const Params& params
);

std::vector<double> func_beta(
    const TrialContext& context,
    const std::vector<std::vector<double>>& qvalue,
    double explor,
    const std::vector<std::string>& system,
    const Params& params
);

double func_gamma(
    const TrialContext& context,
    double reward,
    const Params& params
);

std::vector<double> func_delta(
    const TrialContext& context,
    const Params& params
);

int func_epsilon(const TrialContext& context, const Params& params);

std::vector<double> func_zeta(
    const TrialContext& context,
    bool is_nb,
    const std::vector<double>& value0,
    const std::vector<double>& values,
    double reward,
    double utility,
    const std::string& system,
    const Params& params
);

}  // namespace multiRL
