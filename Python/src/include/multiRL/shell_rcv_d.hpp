#pragma once

#include <multiRL/task_sampler.hpp>

#include <string>
#include <vector>

namespace multiRL {

struct RecoveryControl {
    int n_draws = 100;
    unsigned int seed = 123;
    int threads = 0;
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
};

struct RecoverySimulationRow {
    std::string generating_model;
    int draw = 0;
    std::string subid;
    int block = 0;
    int trial = 0;
    std::vector<std::string> object;
    std::vector<double> reward_table;
    std::string action;
    std::string latent;
    std::string simulation;
    std::string position;
    double reward = missing_real();
    std::vector<double> probability;
};

struct RecoveryTruthRow {
    std::string generating_model;
    int draw = 0;
    std::string subid;
    std::vector<double> params;
};

struct RecoveryResult {
    std::vector<std::string> parameter_names;
    std::vector<std::string> cue_names;
    std::vector<RecoverySimulationRow> simulation;
    std::vector<RecoveryTruthRow> truth;
    RecoveryControl control;
};

RecoveryResult shell_rcv_d(
    const RunTask& task,
    const RecoveryControl& control = RecoveryControl(),
    const std::string& generating_model = "model"
);

}  // namespace multiRL
