#pragma once

#include <multiRL/estimate_mle.hpp>

#include <vector>

#ifdef MULTIRL_HAS_NLOPT
#include <nlopt.hpp>
#endif

namespace multiRL {

void update_free_values(
    Params& params,
    const std::vector<double>& free_values
);

std::vector<double> extract_free_values(const Params& params);

CriterionResult evaluate_mle_task(
    const RunTask& task,
    const std::vector<double>& free_values
);

double estimate_objective_value(
    const RunTask& task,
    const std::vector<double>& free_values
);

#ifdef MULTIRL_HAS_NLOPT

double nlopt_mle_objective(
    const std::vector<double>& x,
    std::vector<double>& grad,
    void* data
);

nlopt::opt build_nlopt_optimizer(
    const NLoptControl& control,
    const std::size_t n_params
);

#endif

}  // namespace multiRL
