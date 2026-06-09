#include <multiRL/modify_control.hpp>

#include <algorithm>
#include <cctype>
#include <thread>

namespace multiRL {

namespace {

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](char item) {
        return static_cast<char>(
            std::tolower(static_cast<unsigned char>(item))
        );
    });
    return value;
}

std::string strip_nlopt_prefix(std::string algorithm) {
    const std::string prefix = "NLOPT_";
    if (algorithm.rfind(prefix, 0) == 0) {
        algorithm.erase(0, prefix.size());
    }
    return algorithm;
}

NLoptControl modify_nlopt_control(const NLoptControl& input) {
    NLoptControl out = input;

    /* ---------------------------------------------------------------------- *
     * NLopt global and local optimizer defaults                              *
     * ---------------------------------------------------------------------- */

    if (out.algorithm.empty()) {
        out.algorithm = "GN_MLSL";
    }
    if (out.local_algorithm.empty()) {
        out.local_algorithm = "LN_BOBYQA";
    }
    if (out.local_xtol_rel <= 0.0) {
        out.local_xtol_rel = 1e-8;
    }

    /* ---------------------------------------------------------------------- *
     * NLopt stopping, random seed, and execution defaults                     *
     * ---------------------------------------------------------------------- */

    if (out.xtol_rel <= 0.0) {
        out.xtol_rel = 1e-6;
    }
    if (out.maxeval <= 0) {
        out.maxeval = 10000;
    }
    if (out.seed < 0) {
        out.seed = 1004;
    }
    if (out.print_level < 0) {
        out.print_level = 0;
    }
    if (out.threads < 0) {
        out.threads = 0;
    }

    return out;
}

}  // namespace

std::string normalize_nlopt_algorithm_name(const std::string& algorithm) {
    if (algorithm.empty()) {
        return "GN_MLSL";
    }

    return strip_nlopt_prefix(algorithm);
}

MLEControl modify_control(
    const MLEControl& input,
    const std::string& estimator
) {
    (void) estimator;

    MLEControl out = input;

    /* ---------------------------------------------------------------------- *
     * MLE uses NLopt as its optimization backend                              *
     * ---------------------------------------------------------------------- */

    out.nlopt = modify_nlopt_control(out.nlopt);
    return out;
}

MAPControl modify_control(
    const MAPControl& input,
    const std::string& estimator
) {
    (void) estimator;

    MAPControl out = input;

    /* ---------------------------------------------------------------------- *
     * MAP first runs MLE, so it owns an embedded MLE control                  *
     * ---------------------------------------------------------------------- */

    out.mle = modify_control(out.mle, "mle");

    /* ---------------------------------------------------------------------- *
     * EM-MAP while-loop control                                               *
     * ---------------------------------------------------------------------- */

    if (out.map_maxiter <= 0) {
        out.map_maxiter = 10;
    }
    if (out.map_tol <= 0.0) {
        out.map_tol = 1e-3;
    }
    if (out.map_patience <= 0) {
        out.map_patience = 10;
    }

    return out;
}

MCMCControl modify_control(
    const MCMCControl& input,
    const std::string& estimator
) {
    (void) estimator;

    MCMCControl out = input;

    /* ---------------------------------------------------------------------- *
     * MCMC sampler control                                                    *
     * ---------------------------------------------------------------------- */

    if (out.algorithm.empty()) {
        out.algorithm = "nuts";
    }
    out.algorithm = to_lower(out.algorithm);
    if (out.chains <= 0) {
        out.chains = 4;
    }
    if (out.warmup < 0) {
        out.warmup = 0;
    }
    if (out.samples <= 0) {
        out.samples = 1000;
    }
    if (out.thin <= 0) {
        out.thin = 1;
    }
    if (out.step_size <= 0.0) {
        out.step_size = 0.05;
    }
    if (out.max_tree_depth <= 0) {
        out.max_tree_depth = 8;
    }
    if (out.target_accept <= 0.0 || out.target_accept >= 1.0) {
        out.target_accept = 0.80;
    }
    if (out.seed < 0) {
        out.seed = 1004;
    }
    if (out.print_level < 0) {
        out.print_level = 0;
    }
    if (out.threads < 0) {
        out.threads = 0;
    }

    return out;
}

ABCControl modify_control(
    const ABCControl& input,
    const std::string& estimator
) {
    (void) estimator;

    ABCControl out = input;

    /* ---------------------------------------------------------------------- *
     * ABC simulation and rejection/regression control                         *
     * ---------------------------------------------------------------------- */

    if (out.seed == 0U) {
        out.seed = 123;
    }
    if (out.scope.empty()) {
        out.scope = "individual";
    }
    out.scope = to_lower(out.scope);
    if (
        out.scope != "individual" &&
        out.scope != "shared" &&
        out.scope != "universal"
    ) {
        out.scope = "individual";
    }
    if (out.threads < 0) {
        out.threads = 0;
    }
    if (out.core <= 0) {
        out.core = out.threads > 0 ? out.threads : 1;
    }
    if (out.threads == 0 && out.core > 1) {
        out.threads = out.core;
    }
    if (out.samples <= 0) {
        out.samples = 1000;
    }
    if (out.sample <= 0) {
        out.sample = out.samples;
    }
    if (out.tol <= 0.0 || out.tol > 1.0) {
        out.tol = 0.1;
    }
    if (out.method.empty()) {
        out.method = "rejection";
    }
    if (out.kernel.empty()) {
        out.kernel = "epanechnikov";
    }
    if (out.reduction.empty()) {
        out.reduction = "none";
    }
    if (out.reduce.empty()) {
        out.reduce = out.reduction;
    }
    if (out.reduction == "none" && out.reduce != "none") {
        out.reduction = out.reduce;
    }
    if (out.n_comp < 0) {
        out.n_comp = 0;
    }
    if (out.fake_block < 0) {
        out.fake_block = 0;
    }
    if (out.print_level < 0) {
        out.print_level = 0;
    }

    return out;
}

RNNControl modify_control(
    const RNNControl& input,
    const std::string& estimator
) {
    (void) estimator;

    RNNControl out = input;

    /* ---------------------------------------------------------------------- *
     * RNN training control for the optional LibTorch backend                  *
     * ---------------------------------------------------------------------- */

    if (out.seed < 0) {
        out.seed = 123;
    }
    if (out.scope.empty()) {
        out.scope = "individual";
    }
    out.scope = to_lower(out.scope);
    if (
        out.scope != "individual" &&
        out.scope != "shared" &&
        out.scope != "universal"
    ) {
        out.scope = "individual";
    }
    if (out.epoch <= 0) {
        out.epoch = 100;
    }
    if (out.epochs <= 0) {
        out.epochs = out.epoch;
    }
    if (out.epoch <= 0) {
        out.epoch = out.epochs;
    }
    if (out.batch_size <= 0) {
        out.batch_size = 32;
    }
    if (out.sample <= 0) {
        out.sample = 100;
    }
    if (out.n_draws <= 0) {
        out.n_draws = out.sample;
    }
    if (out.sample <= 0) {
        out.sample = out.n_draws;
    }
    if (out.threads <= 0) {
        const unsigned int hardware_threads =
            std::thread::hardware_concurrency();
        out.threads = hardware_threads > 0U ?
            static_cast<int>(hardware_threads) : 1;
    }
    if (out.interop_threads < 0) {
        out.interop_threads = 0;
    }
    if (out.units <= 0) {
        out.units = 32;
    }
    if (out.layers <= 0) {
        out.layers = 1;
    }
    if (out.learning_rate <= 0.0) {
        out.learning_rate = 0.001;
    }
    if (out.dropout < 0.0 || out.dropout >= 1.0) {
        out.dropout = 0.0;
    }
    if (out.validation_split < 0.0 || out.validation_split >= 1.0) {
        out.validation_split = 0.0;
    }
    if (out.backend.empty()) {
        out.backend = "torch";
    }
    out.backend = to_lower(out.backend);
    if (out.optimizer.empty()) {
        out.optimizer = "adam";
    }
    out.optimizer = to_lower(out.optimizer);
    if (out.loss.empty()) {
        out.loss = "mse";
    }
    out.loss = to_lower(out.loss);
    if (out.architecture.empty()) {
        out.architecture = "gru";
    }
    out.architecture = to_lower(out.architecture);
    if (out.regularization.empty()) {
        out.regularization = "none";
    }
    out.regularization = to_lower(out.regularization);
    if (out.penalty < 0.0) {
        out.penalty = 0.0;
    }
    if (out.device.empty()) {
        out.device = "cpu";
    }
    out.device = to_lower(out.device);
    if (out.device == "cuda") {
        out.device = "gpu";
    }
    if (out.device != "cpu" && out.device != "gpu") {
        out.device = "cpu";
    }

    return out;
}

}  // namespace multiRL
