#pragma once

#include <multiRL/estimate_mle.hpp>

#include <string>
#include <vector>

namespace multiRL {

struct MAPControl {
    MLEControl mle;
    int map_maxiter = 10;
    double map_tol = 1e-3;
    int map_patience = 10;
    bool init_mle = true;
};

struct MCMCControl {
    std::string algorithm = "nuts";
    int chains = 4;
    int warmup = 1000;
    int samples = 1000;
    int thin = 1;
    int leapfrog_steps = 10;
    double step_size = 0.05;
    int max_tree_depth = 8;
    double max_delta_energy = 1000.0;
    double target_accept = 0.80;
    double min_step_size = 1e-8;
    double max_step_size = 1.0;
    double initial_jitter = 1e-6;
    bool adapt_step_size = true;
    long seed = 1004;
    int print_level = 1;
    int threads = 0;
    int progress_refresh_ms = 200;
    double progress_line_interval_sec = 10.0;
    double progress_line_interval_pct = 10.0;
    std::string progress = "bar";
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
};

struct ABCControl {
    std::string scope = "individual";
    int samples = 1000;
    double tol = 0.1;
    std::string method = "rejection";
    std::string kernel = "epanechnikov";
    std::string reduction = "none";
    std::string reduce = "none";
    int n_comp = 0;
    int fake_block = 0;
    bool normalize = true;
    unsigned int seed = 123;
    int threads = 0;
    int print_level = 1;
    int core = 1;
    int sample = 100;
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
};

struct RNNControl {
    std::string scope = "individual";
    int seed = 123;
    std::string device = "cpu";
    int epoch = 100;
    int epochs = 100;
    int batch_size = 32;
    int sample = 100;
    int n_draws = 100;
    int threads = 0;
    int interop_threads = 0;
    int units = 32;
    int layers = 1;
    int subject_embedding_size = 8;
    int verbose = 0;
    double validation_split = 0.0;
    double learning_rate = 0.001;
    double dropout = 0.0;
    std::string backend = "torch";
    std::string optimizer = "adam";
    std::string loss = "mse";
    std::string architecture = "gru";
    std::string regularization = "none";
    double penalty = 0.0;
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
};

MLEControl modify_control(
    const MLEControl& input,
    const std::string& estimator
);

MAPControl modify_control(
    const MAPControl& input,
    const std::string& estimator
);

MCMCControl modify_control(
    const MCMCControl& input,
    const std::string& estimator
);

ABCControl modify_control(
    const ABCControl& input,
    const std::string& estimator
);

RNNControl modify_control(
    const RNNControl& input,
    const std::string& estimator
);

std::string normalize_nlopt_algorithm_name(const std::string& algorithm);

}  // namespace multiRL
