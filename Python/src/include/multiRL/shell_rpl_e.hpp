#pragma once

#include <multiRL/types.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace multiRL {

struct ReplayFit {
    std::string model = "model";
    std::string model_id = "model_1";
    std::string subid = "";
    std::unordered_map<std::string, double> params;
};

struct ReplayRow {
    std::string source = "model";
    std::string model = "";
    std::string model_id = "";
    std::string subid = "";
    int block = 0;
    int trial = 0;
    std::string action = "";
    double reward = missing_real();
};

struct ReplayPlotRow {
    std::string source = "";
    std::string model = "";
    std::string model_id = "";
    std::string subid = "";
    int block = 0;
    std::string action = "";
    int count = 0;
    int n = 0;
    double ratio = missing_real();
};

struct ReplayResult {
    std::vector<ReplayRow> replay;
    std::vector<ReplayPlotRow> plot_data;
    int n_models = 0;
    int n_subjects = 0;
    bool generate = true;
    bool replay_success = true;
};

ReplayResult shell_rpl_e(
    const std::vector<RunTask>& tasks,
    const std::vector<ReplayFit>& fits
);

}  // namespace multiRL
