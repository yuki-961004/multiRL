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

struct ShellFitPModel {
    std::string model;
    std::string model_id;
    std::vector<RunTask> tasks;
};

struct ShellFitPModelResult {
    std::string model;
    std::string model_id;
    ShellFitPResult result;
};

ShellFitPResult shell_fit_p(
    const std::vector<RunTask>& tasks,
    const ShellFitPControl& control = ShellFitPControl()
);

std::vector<ShellFitPModelResult> shell_fit_p(
    const std::vector<ShellFitPModel>& models,
    const ShellFitPControl& control = ShellFitPControl()
);

}  // namespace multiRL
