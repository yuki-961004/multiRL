#pragma once

#include <multiRL/estimate_mle.hpp>

#include <string>

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
    double step_size = 0.05;
    int max_tree_depth = 8;
    double target_accept = 0.80;
    long seed = 1004;
    int print_level = 1;
    int threads = 0;
};

struct ABCControl {
    int seed = 123;
    int core = 1;
    int sample = 100;
    int samples = 1000;
    double tol = 0.1;
    int print_level = 1;
    std::string method = "rejection";
    std::string kernel = "epanechnikov";
    std::string reduction = "none";
};

struct RNNControl {
    int seed = 123;
    int epoch = 100;
    int batch_size = 32;
    int sample = 100;
    int verbose = 0;
    double validation_split = 0.0;
    std::string backend = "keras3";
    std::string optimizer = "adam";
    std::string loss = "categorical_crossentropy";
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
