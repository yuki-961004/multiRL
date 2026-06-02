#ifdef MULTIRL_HAS_STAN

#include <multiRL/algorithm_stan.hpp>

#include <multiRL/process_MDP_free.hpp>

#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/prim/functor/finite_diff_gradient_auto.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

/* ========================================================================== *
 *                              Numeric Helpers
 * ========================================================================== */

// Map probability from (0, 1) to unbounded real line (-inf, +inf).
// Formula: log(p) - log(1-p) = log(p / (1-p)).
// Used in MCMC to remove constraints from bounded variables so sampling
// can proceed in unrestricted space.
double logit(double probability) {
    return std::log(probability) - std::log1p(-probability);
}

// Inverse of logit (Sigmoid function): map unbounded real to (0, 1).
// Formula: 1 / (1 + exp(-x)).
// Numerically stable piecewise handling avoids overflow.
double inv_logit(double value) {
    if (value >= 0.0) {
        const double exp_neg = std::exp(-value);
        return 1.0 / (1.0 + exp_neg);
    }

    const double exp_pos = std::exp(value);
    return exp_pos / (1.0 + exp_pos);
}

// Compute log(inv_logit(x)), i.e. log(1 / (1 + exp(-x))).
// Numerically stable expansion avoids precision loss.
double log_inv_logit(double value) {
    if (value >= 0.0) {
        return -std::log1p(std::exp(-value));
    }
    return value - std::log1p(std::exp(value));
}

// Compute log(1 - inv_logit(x)), used for Jacobian adjustment terms
// when mapping bounded parameters.
double log1m_inv_logit(double value) {
    if (value >= 0.0) {
        return -value - std::log1p(std::exp(-value));
    }
    return -std::log1p(std::exp(value));
}

// Safely clamp a probability value to [eps, 1-eps] to avoid Inf/NaN
// when fed into logit().
double clamp_probability(double value) {
    const double eps = 1e-12;
    return std::max(eps, std::min(value, 1.0 - eps));
}

// Check whether a floating-point value is finite (not NaN or Inf).
bool is_finite(double value) {
    return std::isfinite(value);
}

// Check whether all elements of a vector are finite real numbers.
// Used in HMC gradient computation to detect anomalies.
bool is_finite_vector(const Eigen::VectorXd& values) {
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values(i))) {
            return false;
        }
    }
    return true;
}

// Convert a log-space Metropolis-Hastings acceptance ratio safely
// into a probability between 0.0 and 1.0.
double safe_accept_probability(double log_accept_ratio) {
    if (!std::isfinite(log_accept_ratio)) {
        return 0.0;
    }
    if (log_accept_ratio >= 0.0) {
        return 1.0;
    }
    return std::exp(log_accept_ratio);
}

// Draw standard normal momentum for each dimension, corresponding to
// a unit mass matrix in the HMC dynamical system.
Eigen::VectorXd draw_momentum(
    Eigen::Index n_dim,
    std::mt19937_64& rng
) {
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::VectorXd momentum(n_dim);

    for (Eigen::Index i = 0; i < n_dim; ++i) {
        momentum(i) = normal(rng);
    }

    return momentum;
}

// Add slight random jitter to initial positions so that multiple chains
// start from slightly different points, reducing risk of identical traces.
void jitter_initial(
    Eigen::VectorXd& initial,
    double jitter,
    std::mt19937_64& rng
) {
    if (jitter <= 0.0) {
        return;
    }

    std::normal_distribution<double> normal(0.0, jitter);

    for (Eigen::Index i = 0; i < initial.size(); ++i) {
        initial(i) += normal(rng);
    }
}

// Draw from Exponential(1) distribution for slice sampling in NUTS.
double exponential(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    double u = uniform(rng);
    while (u <= 0.0 || u >= 1.0) {
        u = uniform(rng);
    }

    return -std::log(u);
}

// Draw from Uniform(0, 1) for accept/reject decisions.
double uniform(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    return uniform(rng);
}

}  // namespace

/* ========================================================================== *
 *                           StanAdapter::Adapter
 * ========================================================================== */

namespace multiRL {
namespace StanAdapter {

Adapter::Adapter(const RunTask& task)
    : posterior_(task.priors),
      lower_bounds_(task.params.free_names.size(),
                    -std::numeric_limits<double>::infinity()),
      upper_bounds_(task.params.free_names.size(),
                    std::numeric_limits<double>::infinity()),
      task_(task) {
}

double Adapter::criterion(const Eigen::VectorXd& unconstrained) const {
    double log_jacobian = 0.0;
    const Eigen::VectorXd constrained = constrain(unconstrained, log_jacobian);

    // Apply constrained values to task params
    RunTask local_task = task_;
    for (Eigen::Index i = 0; i < constrained.size(); ++i) {
        const std::string& name = task_.params.free_names[
            static_cast<std::size_t>(i)
        ];
        local_task.params.values[name] = constrained(i);
    }

    const RunResult result = process_MDP_free(local_task);

    const CriterionResult metric = posterior_.evaluate(
        local_task,
        result.result
    );

    ++n_evals_;

    if (!std::isfinite(metric.log_posterior)) {
        return -std::numeric_limits<double>::infinity();
    }

    return metric.log_posterior + log_jacobian;
}

void Adapter::gradient(
    const Eigen::VectorXd& unconstrained,
    double& log_prob,
    Eigen::VectorXd& gradient
) const {
    // Stan Math finite-difference gradient using criterion as target
    auto target = [this](
        const Eigen::VectorXd& x
    ) -> double {
        return this->criterion(x);
    };

    stan::math::finite_diff_gradient_auto(target, unconstrained,
                                          log_prob, gradient);
}

Eigen::VectorXd Adapter::constrain(
    const Eigen::VectorXd& unconstrained,
    double& log_jacobian
) const {
    const Eigen::Index n = unconstrained.size();
    Eigen::VectorXd constrained(n);
    log_jacobian = 0.0;

    for (Eigen::Index i = 0; i < n; ++i) {
        const double z = unconstrained(i);
        const double lb = lower_bounds_[static_cast<std::size_t>(i)];
        const double ub = upper_bounds_[static_cast<std::size_t>(i)];

        // Both finite: use logit-based mapping to [lb, ub]
        if (std::isfinite(lb) && std::isfinite(ub)) {
            const double p = inv_logit(z);
            constrained(i) = lb + (ub - lb) * p;
            log_jacobian += std::log(ub - lb) +
                log_inv_logit(z) + log1m_inv_logit(z);
        }
        // Only lower bound: x = lb + exp(z)
        else if (std::isfinite(lb)) {
            constrained(i) = lb + std::exp(z);
            log_jacobian += z;
        }
        // Only upper bound: x = ub - exp(z)
        else if (std::isfinite(ub)) {
            constrained(i) = ub - std::exp(z);
            log_jacobian += z;
        }
        // Unbounded: identity mapping, no Jacobian adjustment
        else {
            constrained(i) = z;
        }
    }

    return constrained;
}

Eigen::VectorXd Adapter::unconstrain(
    const std::vector<double>& constrained
) const {
    const Eigen::Index n = static_cast<Eigen::Index>(constrained.size());
    Eigen::VectorXd unconstrained(n);

    for (Eigen::Index i = 0; i < n; ++i) {
        const double x = constrained[static_cast<std::size_t>(i)];
        const double lb = lower_bounds_[static_cast<std::size_t>(i)];
        const double ub = upper_bounds_[static_cast<std::size_t>(i)];

        // Both finite: use logit-based inverse mapping
        if (std::isfinite(lb) && std::isfinite(ub)) {
            const double p = clamp_probability(
                (x - lb) / (ub - lb)
            );
            unconstrained(i) = logit(p);
        }
        // Only lower bound: z = log(x - lb)
        else if (std::isfinite(lb)) {
            unconstrained(i) = std::log(std::max(x - lb, 1e-12));
        }
        // Only upper bound: z = log(ub - x)
        else if (std::isfinite(ub)) {
            unconstrained(i) = std::log(std::max(ub - x, 1e-12));
        }
        // Unbounded: identity
        else {
            unconstrained(i) = x;
        }
    }

    return unconstrained;
}

std::vector<double> Adapter::constrain_to_vector(
    const Eigen::VectorXd& unconstrained
) const {
    double log_jacobian = 0.0;
    const Eigen::VectorXd constrained = constrain(
        unconstrained,
        log_jacobian
    );

    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(constrained.size()));
    for (Eigen::Index i = 0; i < constrained.size(); ++i) {
        out.push_back(constrained(i));
    }

    return out;
}

int Adapter::n_evals() const {
    return n_evals_;
}

void sanitize_initial_point(
    std::vector<double>& initial,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    double epsilon
) {
    for (std::size_t i = 0; i < initial.size(); ++i) {
        const double lb = lower_bounds[i];
        const double ub = upper_bounds[i];

        if (std::isfinite(lb) && initial[i] <= lb) {
            initial[i] = lb + epsilon;
        }
        if (std::isfinite(ub) && initial[i] >= ub) {
            initial[i] = ub - epsilon;
        }
    }
}

}  // namespace StanAdapter

/* ========================================================================== *
 *                           Static HMC Runner
 * ========================================================================== */

namespace HMC {
namespace {

// A single step of the leapfrog integrator, updating position and momentum.
void leapfrog_step(
    const StanAdapter::Adapter& adapter,
    Eigen::VectorXd& position,
    Eigen::VectorXd& momentum,
    double step_size
) {
    double log_prob = 0.0;
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(position.size());

    // Half-step momentum
    adapter.gradient(position, log_prob, gradient);
    if (!is_finite_vector(gradient)) {
        gradient.setZero();
    }
    momentum -= 0.5 * step_size * gradient;

    // Full-step position
    position += step_size * momentum;

    // Half-step momentum
    adapter.gradient(position, log_prob, gradient);
    if (!is_finite_vector(gradient)) {
        gradient.setZero();
    }
    momentum -= 0.5 * step_size * gradient;
}

// Run a single static HMC trajectory with L leapfrog steps.
// Returns false if the proposal is rejected, true if accepted.
bool hmc_trajectory(
    const StanAdapter::Adapter& adapter,
    Eigen::VectorXd& current_position,
    double& current_log_prob,
    int leapfrog_steps,
    double step_size,
    double max_delta_energy,
    std::mt19937_64& rng,
    int& n_accept,
    int& n_proposals
) {
    Eigen::VectorXd proposal_position = current_position;
    Eigen::VectorXd proposal_momentum = draw_momentum(
        proposal_position.size(),
        rng
    );

    double initial_energy = 0.0;
    Eigen::VectorXd current_gradient = Eigen::VectorXd::Zero(
        proposal_position.size()
    );
    adapter.gradient(
        proposal_position,
        current_log_prob,
        current_gradient
    );
    initial_energy = -current_log_prob +
        0.5 * proposal_momentum.squaredNorm();

    Eigen::VectorXd momentum = proposal_momentum;
    for (int step = 0; step < leapfrog_steps; ++step) {
        leapfrog_step(adapter, proposal_position, momentum, step_size);
    }

    double proposal_log_prob = 0.0;
    Eigen::VectorXd proposal_gradient = Eigen::VectorXd::Zero(
        proposal_position.size()
    );
    adapter.gradient(
        proposal_position,
        proposal_log_prob,
        proposal_gradient
    );

    if (!std::isfinite(proposal_log_prob)) {
        ++n_proposals;
        return false;
    }

    double final_energy = -proposal_log_prob +
        0.5 * momentum.squaredNorm();
    double log_accept = initial_energy - final_energy;

    // Check energy conservation
    if (std::abs(initial_energy - final_energy) > max_delta_energy) {
        ++n_proposals;
        return false;
    }

    ++n_proposals;

    if (uniform(rng) <= safe_accept_probability(log_accept)) {
        current_position = proposal_position;
        current_log_prob = proposal_log_prob;
        ++n_accept;
        return true;
    }

    return false;
}

}  // namespace

HMCSamplerResult run_chain(
    const StanAdapter::Adapter& adapter,
    const Eigen::VectorXd& initial_unconstrained,
    const MCMCControl& control,
    int chain_id
) {
    HMCSamplerResult out;

    const Eigen::Index n_dim = initial_unconstrained.size();
    if (n_dim <= 0) {
        out.status = -1;
        out.result_message = "Empty parameter space.";
        out.stop_reason = "empty_parameters";
        return out;
    }

    Eigen::VectorXd current_position = initial_unconstrained;
    double current_log_prob = 0.0;
    Eigen::VectorXd current_gradient = Eigen::VectorXd::Zero(n_dim);
    adapter.gradient(current_position, current_log_prob, current_gradient);

    if (!std::isfinite(current_log_prob)) {
        out.status = -1;
        out.result_message = "Invalid initial posterior.";
        out.stop_reason = "invalid_initial_state";
        return out;
    }

    std::mt19937_64 rng(
        static_cast<uint64_t>(control.seed) +
        static_cast<uint64_t>(chain_id)
    );

    double step_size = control.step_size;
    const int total_iterations = control.warmup + control.samples;
    out.draws.reserve(static_cast<size_t>(control.samples));
    out.log_prob.reserve(static_cast<size_t>(control.samples));

    for (int iter = 0; iter < total_iterations; ++iter) {
        hmc_trajectory(
            adapter,
            current_position,
            current_log_prob,
            control.leapfrog_steps,
            step_size,
            control.max_delta_energy,
            rng,
            out.n_accept,
            out.n_proposals
        );

        if (iter >= control.warmup) {
            out.draws.push_back(
                adapter.constrain_to_vector(current_position)
            );
            out.log_prob.push_back(current_log_prob);
        }
    }

    out.final_step_size = step_size;
    if (out.n_proposals > 0) {
        out.accept_rate =
            static_cast<double>(out.n_accept) /
            static_cast<double>(out.n_proposals);
    }

    if (static_cast<int>(out.draws.size()) == control.samples) {
        out.status = 1;
        out.result_message = "HMC sampling finished.";
        out.stop_reason = "complete";
    } else {
        out.status = -1;
        out.result_message = "HMC produced fewer draws than expected.";
        out.stop_reason = "insufficient_draws";
    }

    return out;
}

}  // namespace HMC

/* ========================================================================== *
 *                             NUTS Runner
 * ========================================================================== */

namespace NUTS {
namespace {

// State of the NUTS sampler during tree doubling.
struct State {
    Eigen::VectorXd position;
    Eigen::VectorXd momentum;
    Eigen::VectorXd gradient;
    double log_prob = 0.0;
    bool valid = false;
};

// Result of building a tree in one direction.
struct TreeResult {
    State left;
    State right;
    State candidate;
    int n_valid = 0;
    bool valid_candidate = false;
    int accept_count = 0;
    double accept_sum = 0.0;
    bool continue_tree = true;
};

// Check U-turn condition between left and right states.
bool no_u_turn(const State& left, const State& right) {
    const Eigen::VectorXd delta = right.position - left.position;

    const double dot_left = delta.dot(left.momentum);
    const double dot_right = delta.dot(right.momentum);

    return (dot_left > 0.0) && (dot_right > 0.0);
}

// Recursively build a binary tree for NUTS trajectory expansion.
TreeResult build_tree(
    const StanAdapter::Adapter& adapter,
    State state,
    int direction,
    int depth,
    double log_slice,
    double initial_joint,
    double step_size,
    double max_delta_energy,
    std::mt19937_64& rng
) {
    if (depth == 0) {
        // Base case: single leapfrog step
        State left = state;
        State right = state;

        double log_prob = 0.0;
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(
            left.position.size()
        );

        // Half-step momentum
        adapter.gradient(left.position, log_prob, gradient);
        if (!is_finite_vector(gradient)) {
            gradient.setZero();
        }
        left.momentum += 0.5 * static_cast<double>(direction) *
            step_size * gradient;
        right.momentum = left.momentum;

        // Full-step position
        left.position += static_cast<double>(direction) * step_size *
            left.momentum;

        // Half-step momentum
        adapter.gradient(left.position, log_prob, gradient);
        if (!is_finite_vector(gradient)) {
            gradient.setZero();
        }
        left.momentum += 0.5 * static_cast<double>(direction) *
            step_size * gradient;

        left.log_prob = log_prob;
        left.gradient = gradient;
        right = left;

        TreeResult out;
        out.left = left;
        out.right = right;

        if (!std::isfinite(log_prob)) {
            out.valid_candidate = false;
            out.n_valid = 0;
            out.continue_tree = false;
            return out;
        }

        double joint = log_prob -
            0.5 * left.momentum.squaredNorm();

        double log_accept = joint - initial_joint;
        if (std::abs(joint - initial_joint) > max_delta_energy) {
            out.valid_candidate = false;
            out.n_valid = 0;
            out.continue_tree = false;
            return out;
        }

        bool accept = (joint > log_slice);
        out.n_valid = accept ? 1 : 0;
        out.valid_candidate = accept;
        out.continue_tree = accept || (joint > log_slice - max_delta_energy);

        if (accept && uniform(rng) <= safe_accept_probability(log_accept)) {
            out.candidate = left;
            out.accept_count = 1;
            out.accept_sum = std::min(1.0, std::exp(log_accept));
        }

        return out;
    }

    // Recursive case: build subtree
    TreeResult sub = build_tree(
        adapter,
        state,
        direction,
        depth - 1,
        log_slice,
        initial_joint,
        step_size,
        max_delta_energy,
        rng
    );

    if (!sub.continue_tree) {
        return sub;
    }

    TreeResult next;
    if (direction == -1) {
        next = build_tree(
            adapter,
            sub.left,
            direction,
            depth - 1,
            log_slice,
            initial_joint,
            step_size,
            max_delta_energy,
            rng
        );
        sub.left = next.left;
    } else {
        next = build_tree(
            adapter,
            sub.right,
            direction,
            depth - 1,
            log_slice,
            initial_joint,
            step_size,
            max_delta_energy,
            rng
        );
        sub.right = next.right;
    }

    const int combined_valid = sub.n_valid + next.n_valid;

    if (next.valid_candidate && combined_valid > 0) {
        const double choose_next =
            static_cast<double>(next.n_valid) /
            static_cast<double>(combined_valid);

        if (uniform(rng) < choose_next) {
            sub.candidate = next.candidate;
        }
    }

    sub.n_valid = combined_valid;
    sub.valid_candidate = sub.valid_candidate || next.valid_candidate;
    sub.accept_count += next.accept_count;
    sub.accept_sum += next.accept_sum;
    sub.continue_tree = next.continue_tree &&
        no_u_turn(sub.left, sub.right);

    return sub;
}

}  // namespace

HMCSamplerResult run_chain(
    const StanAdapter::Adapter& adapter,
    const Eigen::VectorXd& initial_unconstrained,
    const MCMCControl& control,
    int chain_id
) {
    HMCSamplerResult out;

    const Eigen::Index n_dim = initial_unconstrained.size();
    if (n_dim <= 0) {
        out.status = -1;
        out.result_message = "Empty parameter space.";
        out.stop_reason = "empty_parameters";
        return out;
    }

    // Initialize current state
    State current;
    current.position = initial_unconstrained;
    current.momentum = Eigen::VectorXd::Zero(n_dim);
    current.gradient = Eigen::VectorXd::Zero(n_dim);

    adapter.gradient(
        current.position,
        current.log_prob,
        current.gradient
    );

    if (!std::isfinite(current.log_prob)) {
        out.status = -1;
        out.result_message = "Invalid initial posterior.";
        out.stop_reason = "invalid_initial_state";
        return out;
    }

    current.valid = true;
    std::mt19937_64 init_rng_nuts(static_cast<uint64_t>(control.seed) + static_cast<uint64_t>(chain_id)); current.momentum = draw_momentum(n_dim, init_rng_nuts);

    std::mt19937_64 rng(
        static_cast<uint64_t>(control.seed) +
        static_cast<uint64_t>(chain_id) + 1000
    );

    double step_size = control.step_size;
    double accept_rate_sum = 0.0;
    const int total_iterations = control.warmup +
        control.samples * control.thin;

    out.draws.reserve(static_cast<size_t>(control.samples));
    out.log_prob.reserve(static_cast<size_t>(control.samples));

    for (int iter = 0; iter < total_iterations; ++iter) {
        current.momentum = draw_momentum(n_dim, rng);
        const double initial_joint =
            current.log_prob - 0.5 * current.momentum.squaredNorm();
        const double log_slice = initial_joint - exponential(rng);

        State left = current;
        State right = current;
        State proposal = current;

        int n_valid = 1;
        int depth = 0;
        bool keep_sampling = true;
        double accept_sum = 0.0;
        int accept_count = 0;

        // Dynamic tree doubling until U-turn or max depth
        while (keep_sampling && depth < control.max_tree_depth) {
            const int direction = (uniform(rng) < 0.5) ? -1 : 1;
            TreeResult tree;

            if (direction == -1) {
                tree = build_tree(
                    adapter,
                    left,
                    direction,
                    depth,
                    log_slice,
                    initial_joint,
                    step_size,
                    control.max_delta_energy,
                    rng
                );
                left = tree.left;
            } else {
                tree = build_tree(
                    adapter,
                    right,
                    direction,
                    depth,
                    log_slice,
                    initial_joint,
                    step_size,
                    control.max_delta_energy,
                    rng
                );
                right = tree.right;
            }

            if (tree.valid_candidate && tree.n_valid > 0) {
                const int combined_valid = n_valid + tree.n_valid;
                const double choose_tree =
                    static_cast<double>(tree.n_valid) /
                    static_cast<double>(combined_valid);

                if (uniform(rng) < choose_tree) {
                    proposal = tree.candidate;
                }
                n_valid = combined_valid;
            }

            accept_sum += tree.accept_sum;
            accept_count += tree.accept_count;
            keep_sampling = tree.continue_tree &&
                no_u_turn(left, right);
            ++depth;
        }

        if (proposal.valid) {
            current.position = proposal.position;
            current.log_prob = proposal.log_prob;
            current.gradient = proposal.gradient;
        }

        const double accept_probability = (accept_count > 0)
            ? accept_sum / static_cast<double>(accept_count)
            : 0.0;
        accept_rate_sum += accept_probability;
        out.n_proposals += 1;

        if (iter < control.warmup && control.adapt_step_size) {
            const double adapt_rate =
                1.0 / std::sqrt(static_cast<double>(iter + 1));
            const double log_step =
                std::log(step_size) +
                adapt_rate * (accept_probability - control.target_accept);
            step_size = std::exp(log_step);
            step_size = std::max(
                control.min_step_size,
                std::min(step_size, control.max_step_size)
            );
        }

        if (iter >= control.warmup) {
            const int sampling_iter = iter - control.warmup;
            if (sampling_iter % control.thin == 0) {
                out.draws.push_back(
                    adapter.constrain_to_vector(current.position)
                );
                out.log_prob.push_back(current.log_prob);
            }
        }
    }

    out.final_step_size = step_size;
    if (out.n_proposals > 0) {
        out.accept_rate =
            accept_rate_sum / static_cast<double>(out.n_proposals);
    }

    if (static_cast<int>(out.draws.size()) == control.samples) {
        out.status = 1;
        out.result_message = "NUTS sampling finished.";
        out.stop_reason = "complete";
    } else {
        out.status = -1;
        out.result_message = "NUTS produced fewer draws than needed.";
        out.stop_reason = "insufficient_draws";
    }

    return out;
}

}  // namespace NUTS

}  // namespace multiRL

#endif // MULTIRL_HAS_STAN
