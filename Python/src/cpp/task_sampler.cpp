#include <multiRL/task_sampler.hpp>

#include <multiRL/process_model_free.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace multiRL {
namespace {

double sampler_finite_or(double value, double fallback) {
    return std::isfinite(value) ? value : fallback;
}

double sampler_beta_param(
    double shape1,
    double shape2,
    std::mt19937& rng
) {
    if (shape1 <= 0.0 || shape2 <= 0.0) {
        return missing_real();
    }

    std::gamma_distribution<double> left(shape1, 1.0);
    std::gamma_distribution<double> right(shape2, 1.0);
    const double x = left(rng);
    const double y = right(rng);
    if (x + y <= 0.0) {
        return missing_real();
    }
    return x / (x + y);
}

double sampler_prior_param(
    const PriorSpec& prior,
    std::mt19937& rng
) {
    switch (prior.type) {
        case PriorType::NORMAL: {
            if (prior.param2 <= 0.0) {
                return missing_real();
            }
            std::normal_distribution<double> dist(
                prior.param1,
                prior.param2
            );
            return dist(rng);
        }
        case PriorType::UNIFORM: {
            if (prior.param2 < prior.param1) {
                return missing_real();
            }
            std::uniform_real_distribution<double> dist(
                prior.param1,
                prior.param2
            );
            return dist(rng);
        }
        case PriorType::LOGNORMAL: {
            if (prior.param2 <= 0.0) {
                return missing_real();
            }
            std::lognormal_distribution<double> dist(
                prior.param1,
                prior.param2
            );
            return dist(rng);
        }
        case PriorType::CAUCHY: {
            if (prior.param2 <= 0.0) {
                return missing_real();
            }
            std::cauchy_distribution<double> dist(
                prior.param1,
                prior.param2
            );
            return dist(rng);
        }
        case PriorType::BETA:
            return sampler_beta_param(prior.param1, prior.param2, rng);
        case PriorType::EXPONENTIAL: {
            if (prior.param1 <= 0.0) {
                return missing_real();
            }
            std::exponential_distribution<double> dist(prior.param1);
            return dist(rng);
        }
        case PriorType::NONE:
            return missing_real();
    }

    return missing_real();
}

bool sampler_in_bounds(
    double value,
    double lower,
    double upper
) {
    if (!std::isfinite(value)) {
        return false;
    }
    if (std::isfinite(lower) && value < lower) {
        return false;
    }
    if (std::isfinite(upper) && value > upper) {
        return false;
    }
    return true;
}

std::string sampler_task_subid(const RunTask& task) {
    if (!task.input.idinfo.empty() && !task.input.idinfo[0].empty()) {
        return task.input.idinfo[0][0];
    }
    return "1";
}

double sampler_param(
    const std::string& name,
    double initial,
    double lower,
    double upper,
    const PriorSpec* prior,
    std::mt19937& rng
) {
    if (prior != nullptr && prior->type != PriorType::NONE) {
        for (int attempt = 0; attempt < 1000; ++attempt) {
            const double value = sampler_prior_param(*prior, rng);
            if (sampler_in_bounds(value, lower, upper)) {
                return value;
            }
        }
    }

    if (!std::isfinite(lower) || !std::isfinite(upper)) {
        if (name == "alpha" || name == "lapse" || name == "weight") {
            lower = 0.0;
            upper = 1.0;
        } else if (name == "beta") {
            lower = 0.0;
            upper = 10.0;
        } else {
            const double span = std::max(std::abs(initial), 1.0);
            lower = initial - span;
            upper = initial + span;
        }
    }

    if (upper < lower) {
        std::swap(lower, upper);
    }
    if (upper == lower) {
        return lower;
    }

    std::uniform_real_distribution<double> dist(lower, upper);
    return dist(rng);
}

RunTask sampled_task(
    const RunTask& task,
    const TaskSamplerControl& control,
    const int draw
) {
    RunTask out = task;
    out.settings.generate = true;
    out.params.values["seed"] =
        static_cast<double>(control.seed + draw + 1);

    std::mt19937 rng(
        static_cast<std::mt19937::result_type>(control.seed + draw + 1001)
    );

    for (std::size_t index = 0; index < out.params.free_names.size();
         ++index) {
        const std::string& name = out.params.free_names[index];
        const double initial = out.params.get(name);
        const double lower = index < control.lower_bounds.size()
            ? control.lower_bounds[index]
            : -std::numeric_limits<double>::infinity();
        const double upper = index < control.upper_bounds.size()
            ? control.upper_bounds[index]
            : std::numeric_limits<double>::infinity();

        const PriorSpec* prior = nullptr;
        if (out.priors.active) {
            const auto found = out.priors.specs.find(name);
            if (found != out.priors.specs.end()) {
                prior = &found->second;
            }
        }

        out.params.values[name] = sampler_param(
            name,
            initial,
            lower,
            upper,
            prior,
            rng
        );
    }

    return out;
}

std::vector<TaskSamplerRow> rows_from_draw(
    const RunTask& task,
    const RunResult& result,
    const int draw
) {
    std::vector<TaskSamplerRow> out;
    out.reserve(task.input.n_rows);
    const std::string subid = sampler_task_subid(task);

    for (std::size_t row = 0; row < task.input.n_rows; ++row) {
        TaskSamplerRow value;
        value.draw = draw + 1;
        value.sequence = static_cast<int>(row);
        value.subid = subid;
        value.block = task.input.block[row];
        value.trial = task.input.trial[row];
        value.action = task.input.action[row];
        value.latent = result.result.latent[row];
        value.simulation = result.result.simulation[row];
        value.position = result.result.position[row];
        value.reward = sampler_finite_or(
            result.result.reward[row],
            missing_real()
        );
        value.probability = result.result.prob[row];

        value.params.reserve(task.params.free_names.size());
        for (const std::string& name : task.params.free_names) {
            value.params.push_back(task.params.get(name));
        }

        out.push_back(value);
    }

    return out;
}

}  // namespace

TaskSamplerResult task_sampler(
    const RunTask& task,
    const TaskSamplerControl& raw_control
) {
    TaskSamplerControl control = raw_control;
    if (control.n_draws < 1) {
        control.n_draws = 1;
    }

    std::vector<std::vector<TaskSamplerRow>> draw_rows(
        static_cast<std::size_t>(control.n_draws)
    );

#ifdef _OPENMP
    if (control.threads > 0) {
        omp_set_num_threads(control.threads);
    }
#pragma omp parallel for schedule(dynamic) if(control.n_draws > 1)
#endif
    for (int draw = 0; draw < control.n_draws; ++draw) {
        RunTask local_task = sampled_task(task, control, draw);
        const RunResult local_result = process_model_free(local_task);
        draw_rows[static_cast<std::size_t>(draw)] = rows_from_draw(
            local_task,
            local_result,
            draw
        );
    }

    TaskSamplerResult out;
    out.parameter_names = task.params.free_names;
    out.cue_names = task.behrule.cue;
    out.control = control;
    out.generate = true;

    std::size_t n_rows = 0;
    for (const auto& value : draw_rows) {
        n_rows += value.size();
    }
    out.rows.reserve(n_rows);

    for (const auto& value : draw_rows) {
        out.rows.insert(out.rows.end(), value.begin(), value.end());
    }

    return out;
}

std::vector<TaskSamplerResult> task_sampler(
    const std::vector<RunTask>& tasks,
    const TaskSamplerControl& raw_control
) {
    TaskSamplerControl control = raw_control;
    if (control.n_draws < 1) {
        control.n_draws = 1;
    }

    std::vector<TaskSamplerResult> out(tasks.size());

#ifdef _OPENMP
    if (control.threads > 0) {
        omp_set_num_threads(control.threads);
    }
#pragma omp parallel for schedule(dynamic) if(tasks.size() > 1)
#endif
    for (int index = 0; index < static_cast<int>(tasks.size()); ++index) {
        out[static_cast<std::size_t>(index)] =
            task_sampler(tasks[static_cast<std::size_t>(index)], control);
        for (TaskSamplerRow& row : out[static_cast<std::size_t>(index)].rows) {
            row.subject_index = index;
        }
    }

    return out;
}

}  // namespace multiRL
