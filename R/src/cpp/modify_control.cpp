#include <multiRL/modify_control.hpp>

#include <algorithm>
#include <cctype>

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
     * RNN training control for keras and keras3 wrappers                      *
     * ---------------------------------------------------------------------- */

    if (out.seed < 0) {
        out.seed = 123;
    }
    if (out.epoch <= 0) {
        out.epoch = 100;
    }
    if (out.batch_size <= 0) {
        out.batch_size = 32;
    }
    if (out.sample <= 0) {
        out.sample = 100;
    }
    if (out.validation_split < 0.0 || out.validation_split >= 1.0) {
        out.validation_split = 0.0;
    }
    if (out.backend.empty()) {
        out.backend = "keras3";
    }
    if (out.optimizer.empty()) {
        out.optimizer = "adam";
    }
    if (out.loss.empty()) {
        out.loss = "categorical_crossentropy";
    }

    return out;
}

}  // namespace multiRL
