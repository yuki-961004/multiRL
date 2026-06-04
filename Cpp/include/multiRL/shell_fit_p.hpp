#pragma once

#include <multiRL/estimate_abc.hpp>
#include <multiRL/estimate_map.hpp>
#include <multiRL/estimate_mcmc.hpp>
#include <multiRL/estimate_mle.hpp>

#include <string>
#include <vector>

namespace multiRL {

struct ShellFitPControl {
    std::string estimator = "mle";
    MLEControl mle;
    MAPControl map;
    MCMCControl mcmc;
    ABCControl abc;
};

struct ShellFitPResult {
    std::string estimator;
    std::vector<EstimateMleResult> mle;
    EstimateMapResult map;
    std::vector<SubjectMCMCResult> mcmc;
    std::vector<ABCSubjectResult> abc;
};

ShellFitPResult shell_fit_p(
    const std::vector<RunTask>& tasks,
    const ShellFitPControl& control = ShellFitPControl()
);

}  // namespace multiRL
